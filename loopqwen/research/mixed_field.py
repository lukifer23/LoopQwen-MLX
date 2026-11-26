"""Run a complete mixed workload through one shipping assistant or untouched Base."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from jsonschema import ValidationError, validate
from mlx_lm.utils import load_tokenizer

from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.mixed_field import catalog, grade
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..package_tools import snapshot
from ..tool_model import Assistant
from ..tool_session import compact, decode_action, initial_messages, prompt
from .core import Budget
from .registered import registered


def outcome(data, spec):
    groups = sorted({r["family"] for r in data["candidate"]})
    results = {}
    for group in ["all", *groups]:
        cells = {}
        filtered = []
        for side, batch in data.items():
            rows = [r for r in batch if group == "all" or r["family"] == group]
            filtered.append(rows)
            cells[side] = dict(
                n=len(rows),
                correct=sum(r["score"]["success"] for r in rows),
                completed=sum(r["score"]["completed"] for r in rows),
                seconds=sum(r["wall_s"] for r in rows),
                peak=max(r["request_peak"] for r in rows),
                first_token_seconds=sum(r["first_token_s"] for r in rows),
                tokens=sum(
                    len(t["token_ids"]) for r in rows for t in r["trajectory"]["turns"]
                ),
            )
        pairs = [
            int(a["score"]["success"]) - int(b["score"]["success"])
            for a, b in zip(*filtered, strict=True)
        ]
        results[group] = dict(
            **cells,
            wins=pairs.count(1),
            losses=pairs.count(-1),
            p=exact_mcnemar(pairs.count(1), pairs.count(-1)),
        )
    pooled = results["all"]
    passed = (
        pooled["wins"] - pooled["losses"] >= spec["net_floor"]
        and pooled["p"] <= 0.05
        and all(
            g["candidate"]["correct"] >= g["control"]["correct"]
            for g in results.values()
        )
        and pooled["candidate"]["seconds"] <= pooled["control"]["seconds"]
        and pooled["candidate"]["peak"] <= pooled["control"]["peak"]
    )
    strict = all(
        g["candidate"]["correct"] > g["control"]["correct"]
        and all(
            g["candidate"][k] < g["control"][k]
            for k in ("seconds", "peak", "first_token_seconds")
        )
        for group, g in results.items()
        if group != "all"
    )
    return dict(passed=passed, strict_field_dominance=strict, by_family=results)


def bundle_for(spec):
    newer = Path(spec["newer_release"])
    prior = Path(spec["requires"])
    registry = read("research/registry.json")["experiments"]
    for p in (newer, prior):
        if digest(p / "checksums.json") != registry[p.name]["archive"]:
            raise ValueError("Unsealed field release")
    if (
        digest(prior / "checksums.json") != spec["prerequisite_sha256"]
        or not read(prior / "decision.json")["passed"]
    ):
        raise ValueError("Missing prior field qualification")
    if digest(newer / "registration.json") != spec["newer_registration"]:
        raise ValueError("Different release selection")
    chosen = newer if read(newer / "decision.json")["passed"] else prior
    bundle = Path(read(chosen / "registration.json")["destination"])
    for path, sha in read(chosen / "export-files.json").items():
        if digest(bundle / path) != sha:
            raise ValueError("Changed release file " + path)
    return chosen, bundle


class DenseAssistant(Assistant):
    """Expose the identical public decoding protocol with one untouched dense model."""

    def __init__(self, reuse):
        self.language, self.tokenizer, _ = base()
        self._tool_model = self.language
        self.reuse_prefix = reuse
        self.dedicated = False


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    parent, bundle = bundle_for(spec)
    material = read(spec["material"])
    if snapshot() != material["environment"]:
        raise ValueError("Local package environment changed")
    for path, reference in material["files"].items():
        if digest(Path(path)) != reference["sha256"]:
            raise ValueError("Changed project file")
    default = bool(read(bundle / "tool/tool-model.json").get("prefix_reuse"))
    save(
        out / "deployment.json",
        dict(
            release=parent.name,
            archive=digest(parent / "checksums.json"),
            prefix=default,
        ),
    )
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        for side in ("candidate", "control"):
            rows = []
            assistant = None
            for i, w in enumerate(material["rows"]):
                budget.check()
                tools = catalog(w, material)
                mx.synchronize()
                mx.reset_peak_memory()
                at = perf_counter()
                if assistant is None:
                    assistant = (
                        Assistant(bundle)
                        if side == "candidate"
                        else DenseAssistant(default)
                    )
                first = []

                def progress(
                    consumed, first=first, length=len(w["initial_ids"]), started=at
                ):
                    if not first and consumed == length:
                        first.append(perf_counter() - started)
                    budget.check()

                tr = assistant.respond(
                    w.get("user_request", w["question"]),
                    tools,
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    progress=progress,
                )
                mx.synchronize()
                wall = perf_counter() - at
                assert first and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                rows.append(
                    dict(
                        first_token_s=first[0],
                        id=w["id"],
                        family=w["family"],
                        trajectory=tr,
                        score=grade(w, tr, assistant.tokenizer),
                        wall_s=wall,
                        request_peak=mx.get_peak_memory(),
                    )
                )
                if i % 10 == 9:
                    print(
                        side,
                        i + 1,
                        sum(r["score"]["success"] for r in rows),
                        flush=True,
                    )
            packed(out / (side + ".json.gz"), rows)
            data[side] = rows
            del assistant
            gc.collect()
            mx.clear_cache()
        d = outcome(data, spec)
        save(out / "decision.json", d)
        print(d, flush=True)
    finally:
        budget.close()


def replay(row, record, material, tok, spec):
    tools = catalog(row, material, True)
    request = row.get("user_request", row["question"])
    messages = (
        initial_messages(request, tools)
        if tools
        else [dict(role="user", content=request)]
    )
    available = {t.name: t for t in tools}
    tr = record["trajectory"]
    stops = tok.eos_token_ids | {tok.convert_tokens_to_ids("<|im_end|>")}
    answer = None
    status = "turn_limit"
    assert (
        tr["turns"][0]["prompt_ids"] == row["initial_ids"]
        and 0 < record["first_token_s"] <= record["wall_s"]
    )
    assert 1 <= len(tr["turns"]) <= (spec["max_turns"] if tools else 1)
    for index, t in enumerate(tr["turns"]):
        assert t["prompt_ids"] == tok.encode(
            prompt(tok, messages), add_special_tokens=False
        )
        ids = t["token_ids"]
        assert 0 < len(ids) <= spec["max_tokens"] and not any(
            i in stops for i in ids[:-1]
        )
        assert 0 < t["ttft_s"] <= t["seconds"]
        if ids[-1] not in stops:
            assert len(ids) == spec["max_tokens"]
            status = "token_limit"
        else:
            raw = tok.decode(ids[:-1])
            _, closed, visible = raw.partition("</think>")
            visible = visible.strip()
            action = None
            if not closed:
                status = "unfinished_thinking"
            else:
                assert visible == t["visible"]
                if tools:
                    try:
                        action = decode_action(visible)
                        if action is not None:
                            fn = available[action["name"]]
                            validate(action["arguments"], fn.parameters)
                    except (KeyError, ValueError, ValidationError):
                        status = "invalid_call"
                        assert index + 1 == len(tr["turns"])
                        break
                if action is None:
                    answer = visible
                    status = "answered" if answer else "empty_answer"
                else:
                    assert action == t["call"]
                    try:
                        obs = fn.callback(**action["arguments"])
                    except Exception as e:
                        obs = {"error": type(e).__name__}
                    assert obs == t["observation"]
                    messages.extend(
                        [
                            dict(role="assistant", content=visible),
                            dict(role="tool", content=compact(obs)),
                        ]
                    )
                    continue
        assert index + 1 == len(tr["turns"])
    assert tr["status"] == status and tr["answer"] == answer
    assert (
        record["score"] == grade(row, tr, tok)
        and record["wall_s"] >= tr["seconds"]
        and record["request_peak"] > 0
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    material = read(spec["material"])
    tok = load_tokenizer(local_snapshot())
    data = {}
    for side in ("candidate", "control"):
        rows = unpack(root / (side + ".json.gz"))
        for w, r in zip(material["rows"], rows, strict=True):
            assert (w["id"], w["family"]) == (r["id"], r["family"])
            replay(w, r, material, tok, spec)
        data[side] = rows
    assert outcome(data, spec) == decision


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
