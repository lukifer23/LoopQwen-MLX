"""Measure a compute-rebalanced full-history register graph at fixed decoder-site count."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..recurrence import Span
from ..identity import fingerprint
from ..model_loader import local_snapshot
from .core import Budget
from .ledger_curriculum import measure
from .registered import registered


class Rebalanced(LoopedModel):
    @property
    def sites(self):
        # Second application at layer 6 has its own delta state. Replace global site 7.
        return tuple(site for site in super().sites if site != (7, 0))


class RegisterDispatch:
    def __init__(self, original, alternative, route):
        self.route = route
        self.original = original
        self.alternative = alternative

    def choose(self, model, ids, cap):
        selected, name = self.original.choose(model, ids, cap)
        if name == "register" and selected is model:
            return self.alternative, self.route
        return selected, name


def attach(model, spec):
    if "local_window" in spec:
        from ..hybrid_attention import HybridAttention

        view = HybridAttention(model.text, spec["local_window"])
        route = "hybrid_register"
    else:
        view = Rebalanced(
            model.text, Span(start=6, length=1, passes=2, mode="stabilized"), alpha=0.35
        )
        assert (
            len(view.sites) == 24
            and sum(not model.text.layers[i].is_linear for i, p in view.sites) == 5
        )
        route = "rebalanced_register"
    object.__setattr__(view, "attention_reader", model.attention_reader)
    view.eval()
    object.__setattr__(
        model, "context_policy", RegisterDispatch(model.context_policy, view, route)
    )
    return view


def outcome(rows, items, spec):
    metrics = {}
    for name, rs in rows.items():
        metrics[name] = dict(
            correct=sum(r["correct"] for r in rs),
            ttft=sum(r["ttft_s"] for r in rs),
            seconds=sum(r["elapsed_s"] for r in rs),
            peak=max(r["peak_bytes"] for r in rs),
        )
    a, b = metrics["candidate"], metrics["control"]
    correct_spans = all(
        r.get("context_route")
        == ("hybrid_register" if "local_window" in spec else "rebalanced_register")
        and r["attention_memory"]["start"] <= w["gold_start"]
        and r["attention_memory"]["stop"] >= w["gold_stop"]
        for r, w in zip(rows["candidate"], items, strict=True)
    )
    passed = (
        correct_spans
        and a["correct"] == len(items)
        and a["correct"] > b["correct"]
        and a["ttft"] <= spec["ttft_ratio"] * b["ttft"]
        and a["seconds"] <= b["seconds"]
        and a["peak"] <= b["peak"]
    )
    if "local_window" in spec:
        for r in rows["candidate"]:
            sites = r.get("attention_sites", [])
            consumed = r["prompt_tokens"] + len(r["token_ids"]) - 1
            global_sites = {
                (s["layer"], s["turn"]) for s in sites if s["global_attention"]
            }
            proof = (
                len(sites) == 7
                and global_sites == {(15, 1), (19, 0), (23, 0)}
                and all(
                    s["offset"] == consumed
                    and s["allocated"] >= s["retained"]
                    and (
                        s["global_attention"]
                        or s["allocated"] <= spec["local_window"] + 127
                    )
                    and (
                        s["retained"] == consumed
                        if s["global_attention"]
                        else 0 < s["retained"] <= spec["local_window"] + 127
                    )
                    for s in sites
                )
            )
            passed = passed and proof
    return dict(passed=passed, metrics=metrics, spans=correct_spans, items=len(items))


def flatten(spec):
    material = read(spec["material"])
    groups = material if isinstance(material, list) else [material]
    return [item for group in groups for item in group["panel"]]


def verify(root, decision):
    from ..evaluation.answers import numeric
    from ..evaluation.language7 import interpret

    spec = read(root / "registration.json")
    items = flatten(spec)
    tok = load_tokenizer(local_snapshot())
    rows = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    for arm in rows.values():
        for r, w in zip(arm, items, strict=True):
            assert r["input_sha256"] == fingerprint(w["prompt_ids"]) and r[
                "prompt_tokens"
            ] == len(w["prompt_ids"])
            ids = r["token_ids"]
            term = {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
            parsed = interpret(
                ids,
                tok.convert_tokens_to_ids("<think>"),
                tok.convert_tokens_to_ids("</think>"),
                term,
                tok,
            )
            ans = numeric(parsed["visible"]) if ids[-1] in term else None
            assert ans == r["answer"] and (ans == w["gold"]) == r["correct"]
    assert outcome(rows, items, spec) == decision


def run(name):
    spec = registered(name)
    items = flatten(spec)
    dest = Path("outputs/research") / name
    if (dest / "decision.json").exists():
        raise ValueError("Existing run")
    parent = Path(spec["requires"])
    assert (
        digest(parent / "checksums.json") == spec["prerequisite_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    budget = Budget(name, spec["limits"])
    rows = {"candidate": [], "control": []}
    try:
        for i, w in enumerate(items):
            order = ("candidate", "control") if i % 2 == 0 else ("control", "candidate")
            for side in order:
                if side == "control":
                    m, t, _ = base()
                else:
                    m, t, _ = load_trained(spec["checkpoint"])
                    view = attach(m, spec)
                mx.reset_peak_memory()
                record = measure(m, t, w, spec["max_tokens"], budget)
                if side == "candidate":
                    if "local_window" in spec:
                        record["attention_sites"] = view.history()
                    del view
                rows[side].append(record)
                print(
                    i,
                    side,
                    record["correct"],
                    record["ttft_s"],
                    record["peak_bytes"],
                    flush=True,
                )
                del m, t
                gc.collect()
                mx.clear_cache()
        for side, arm in rows.items():
            packed(dest / (side + ".json.gz"), arm)
        save(dest / "decision.json", outcome(rows, items, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
