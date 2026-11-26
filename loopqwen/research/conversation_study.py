"""Compare grounded two-user-turn conversations with live package inventory."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..evaluation import conversation_questions, module_questions
from ..evaluation.conversation_questions import score
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..package_tools import snapshot
from ..tool_model import Assistant
from ..tool_session import initial_messages
from .core import Budget
from .mixed_field import DenseAssistant, replay_exchange
from .registered import registered


class FittedConversation(Assistant):
    """Expose two equally fitted checkpoints through the same conversation API."""

    def __init__(self, source):
        from .tool_fit import restore

        self._tool_model, self.tokenizer = restore(source)
        self.reuse_prefix = False
        self.dedicated = True


def matched_sources(spec):
    roots = [Path(s) for s in spec["trained_arms"]]
    for root, h in zip(roots, spec["arm_hashes"], strict=True):
        if digest(root / "checksums.json") != h:
            raise ValueError("Changed checkpoint archive")
    fits = [read(root / "fit.json") for root in roots]
    rules = [read(root / "registration.json") for root in roots]
    if (
        [f["passes"] for f in fits] != [2, 1]
        or any(
            fits[0][k] != fits[1][k]
            for k in ("training", "trainable", "updates", "targets", "turns")
        )
        or any(
            rules[0][k] != rules[1][k]
            for k in (
                "donor",
                "prefix_bits",
                "fit_seed",
                "rank",
                "batch",
                "updates",
                "lr",
            )
        )
    ):
        raise ValueError("Mismatched supervised depth control")
    return dict(zip(("candidate", "control"), roots, strict=True))


def trajectory_ids(rows):
    return [
        [
            [
                (t["prompt_ids"], t["token_ids"], t.get("call"), t.get("observation"))
                for t in exchange["turns"]
            ]
            for exchange in r["exchanges"]
        ]
        for r in rows
    ]


def outcome(data, spec=None):
    metrics = {}
    for name, rows in data.items():
        metrics[name] = {
            "success": sum(r["score"]["success"] for r in rows),
            "first": sum(r["score"]["first"] for r in rows),
            "content": sum(r["score"]["content"] for r in rows),
            "families": {
                k: sum(r["score"]["success"] for r in rows if r["kind"] == k)
                for k in ("reference", "clarification")
            },
            "seconds": sum(r["wall_s"] for r in rows),
            "peak": max(r["peak"] for r in rows),
        }
    diffs = [
        int(a["score"]["success"]) - int(b["score"]["success"])
        for a, b in zip(data["candidate"], data["control"], strict=True)
    ]
    p = exact_mcnemar(diffs.count(1), diffs.count(-1))
    a, b = metrics["candidate"], metrics["control"]
    summary = {
        "passed": sum(diffs) >= 6
        and p <= 0.05
        and min(a["families"].values()) >= 4
        and a["seconds"] <= b["seconds"]
        and a["peak"] <= b["peak"],
        "metrics": metrics,
        "gains": diffs.count(1),
        "losses": diffs.count(-1),
        "p_exact": p,
    }
    if spec and spec.get("trained_arms"):
        reference = unpack(Path(spec["requires"]) / "candidate.json.gz")
        summary["qualified_candidate_replayed"] = trajectory_ids(
            reference
        ) == trajectory_ids(data["candidate"])
        summary["passed"] = (
            summary["qualified_candidate_replayed"]
            and sum(diffs) >= spec["net_floor"]
            and p <= 0.05
            and min(a["families"].values()) >= 4
            and a["seconds"] <= spec["time_ratio"] * b["seconds"]
            and a["peak"] <= spec["peak_ratio"] * b["peak"]
        )
    return summary


def verify(root, decision):
    spec = read(root / "registration.json")
    material = read(spec["material"])
    tok = load_tokenizer(local_snapshot())
    data = {}
    provider = (
        module_questions if spec.get("resource") == "module" else conversation_questions
    )
    tools = provider.catalog(material["environment"])
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for r, w in zip(data[side], material["rows"], strict=True):
            history = initial_messages(w["first"], tools)[:1]
            for q, t in zip((w["first"], w["reply"]), r["exchanges"], strict=True):
                history.append({"role": "user", "content": q})
                replay_exchange(tok, tools, history, t, spec)
            assert (
                history == r["messages"]
                and r["score"]
                == score(w, r["exchanges"], literal_prefix=spec.get("resource") is None)
                and r["id"] == w["id"]
                and r["kind"] == w["kind"]
                and r["exchanges"][0]["turns"][0]["prompt_ids"] == w["initial_ids"]
                and r["wall_s"] >= sum(x["seconds"] for x in r["exchanges"])
                and r["peak"] > 0
            )
    if spec.get("trained_arms"):
        matched_sources(spec)
    assert outcome(data, spec) == decision


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    parent = Path(spec["requires"])
    material = read(spec["material"])
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    if (
        not read(parent / "decision.json")["passed"]
        or digest(parent / "checksums.json") != spec["prerequisite_sha256"]
    ):
        raise ValueError("Unqualified source")
    fitted = matched_sources(spec) if spec.get("trained_arms") else None
    bundle = Path(spec["checkpoint"]) if not fitted else None
    provider = (
        module_questions if spec.get("resource") == "module" else conversation_questions
    )
    environment = (
        module_questions.snapshot() if spec.get("resource") == "module" else snapshot()
    )
    if environment != material["environment"] or (
        bundle is not None
        and any(
            digest(bundle / k) != v
            for k, v in read(parent / "export-files.json").items()
        )
    ):
        raise ValueError("Changed package inventory or deployed weights")
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        for side in ("candidate", "control"):
            assistant = FittedConversation(fitted[side]) if fitted else None
            rs = []
            for w in material["rows"]:
                mx.synchronize()
                mx.reset_peak_memory()
                start = perf_counter()
                if assistant is None:
                    assistant = (
                        Assistant(bundle)
                        if side == "candidate"
                        else DenseAssistant(False)
                    )
                chat = assistant.conversation(provider.catalog())
                xs = [
                    chat.respond(
                        text,
                        max_tokens=spec["max_tokens"],
                        max_turns=spec["max_turns"],
                        progress=lambda _: budget.check(),
                    )
                    for text in (w["first"], w["reply"])
                ]
                mx.synchronize()
                wall = perf_counter() - start
                assert xs[0]["turns"][0]["prompt_ids"] == w["initial_ids"]
                rs.append(
                    {
                        "id": w["id"],
                        "kind": w["kind"],
                        "exchanges": xs,
                        "messages": chat.messages,
                        "score": score(
                            w, xs, literal_prefix=spec.get("resource") is None
                        ),
                        "wall_s": wall,
                        "peak": mx.get_peak_memory(),
                    }
                )
                print(side, w["id"], rs[-1]["score"], flush=True)
                del chat
            data[side] = rs
            packed(root / (side + ".json.gz"), rs)
            del assistant
            gc.collect()
            mx.clear_cache()
        save(root / "decision.json", outcome(data, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
