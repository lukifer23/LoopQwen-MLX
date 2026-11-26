"""Time the emitted tokens in the complete 131K offline decoder under a two-arm crossover."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx

from ..evaluation.answers import numeric
from ..evaluation.language7 import interpret
from ..evaluation.mechanism import packed, unpack
from ..export_runtime import load_export
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from .balanced_register import flatten
from .core import Budget, atom
from .ledger_curriculum import measure
from .register_allocation import packed_evidence
from .registered import registered


def performance(rs):
    elapsed = sum(r["elapsed_s"] for r in rs)
    first = sum(r["ttft_s"] for r in rs)
    steps = sum(len(r["token_ids"]) - 1 for r in rs)
    return {
        "correct": sum(r["correct"] for r in rs),
        "queries": len(rs),
        "steps": steps,
        "seconds": elapsed,
        "ttft": first,
        "decode_seconds": elapsed - first,
        "decode_tps": steps / (elapsed - first),
        "prefill_tps": sum(r["prompt_tokens"] for r in rs) / first,
        "peak": max(r["peak_bytes"] for r in rs),
    }


def outcome(spec, records, items, expected):
    by = {s: [] for s in ("fused", "stock")}
    exact = physical = True
    sites = {(i, 0) for i in (3, 7, 11, 15, 19, 23)} | {(11, 1), (15, 1)}
    repeats = [{s: [] for s in by} for _ in range(2)]
    by_item = [{s: [] for s in by} for w in items]
    for seq, (rec, (side, ix, rep)) in enumerate(
        zip(records, spec["order"], strict=True)
    ):
        assert rec["sequence"] == seq and (rec["side"], rec["item"], rec["repeat"]) == (
            side,
            ix,
            rep,
        )
        r = rec["result"]
        old, w = expected[ix], items[ix]
        n = len(w["prompt_ids"])
        steps = len(r["token_ids"]) - 1
        assert (
            0 < r["ttft_s"] < r["elapsed_s"]
            and r["peak_bytes"] > 0
            and 0 < steps < spec["max_tokens"]
        )
        exact &= (
            r["correct"]
            and r["token_ids"] == old["token_ids"]
            and r["attention_memory"] == old["attention_memory"]
            and r["context_route"] == "register"
        )
        assert r["prompt_tokens"] == n >= 131072 and r["input_sha256"] == old[
            "input_sha256"
        ] == fingerprint(w["prompt_ids"])
        physical &= {(a["layer"], a["turn"]) for a in r["kv_history"]} == sites and all(
            a["offset"] == a["retained"] == n + steps for a in r["kv_history"]
        )
        physical &= {(a["layer"], a["turn"]) for a in r["page_sites"]} == sites and all(
            a["offset"] == a["retained"] == n + steps
            and a["sparse_blocks"] * 128 + a["dense_queries"] == n + steps
            and a["possible"] == (n + steps) * (n + steps + 1) // 2
            and 0 < a["attended"] < a["possible"]
            and packed_evidence(a, n + steps, capacity=n + spec["max_tokens"] - 1)
            and a.get("fused_queries", 0) == (steps if side == "fused" else 0)
            for a in r["page_sites"]
        )
        by[side].append(r)
        repeats[rep][side].append(r)
        by_item[ix][side].append(r)

    def metrics(subsets):
        return [{s: performance(rs) for s, rs in sub.items()} for sub in subsets]

    total = metrics([by])[0]
    timing = metrics(repeats)
    docs = metrics(by_item)
    a, b = total["fused"], total["stock"]
    ratio = a["decode_tps"] / b["decode_tps"]
    passed = (
        exact
        and physical
        and ratio >= spec["min_speedup"]
        and all(
            pair["fused"]["decode_tps"] > pair["stock"]["decode_tps"]
            for pair in timing + docs
        )
        and all(a[k] <= b[k] * spec["timing_tolerance"] for k in ("ttft", "seconds"))
        and a["peak"] <= b["peak"] * spec["peak_tolerance"]
    )
    return {
        "passed": bool(passed),
        "identical_correct": bool(exact),
        "physical": bool(physical),
        "speedup": ratio,
        "metrics": total,
        "repeats": timing,
        "documents": docs,
    }


def verify(root, d):
    from mlx_lm.utils import load_tokenizer

    spec = read(root / "registration.json")
    items = flatten(spec)
    records = unpack(root / "runs.json.gz")
    old = unpack(Path(spec["published"]) / "direct_pages.json.gz")
    tok = load_tokenizer(local_snapshot())
    eos = {*tok.eos_token_ids, atom(tok, "<|im_end|>")}
    for e in records:
        r = e["result"]
        v = interpret(
            r["token_ids"], atom(tok, "<think>"), atom(tok, "</think>"), eos, tok
        )
        answer = numeric(v["visible"]) if r["token_ids"][-1] in eos else None
        assert (
            r["gold"] == items[e["item"]]["gold"]
            and r["answer"] == answer
            and r["correct"] == (answer == r["gold"])
        )
    assert d == outcome(spec, records, items, old)


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    for k in ("parent", "release", "published"):
        src = Path(spec[k])
        assert (
            digest(src / "checksums.json") == spec[k + "_sha256"]
            and read(src / "decision.json")["passed"]
        )
    items = flatten(spec)
    old = unpack(Path(spec["published"]) / "direct_pages.json.gz")
    src = Path(spec["checkpoint"])
    ref = read(Path(spec["release"]) / "export-files.json")
    assert all(digest(src / k) == v for k, v in ref.items())
    dest = out / "runs.json.gz"
    records = unpack(dest) if dest.exists() else []
    assert all(
        (e["side"], e["item"], e["repeat"]) == tuple(spec["order"][i])
        for i, e in enumerate(records)
    )
    budget = Budget(name, spec["limits"])
    try:
        for i, (side, ix, rep) in enumerate(spec["order"]):
            if i < len(records):
                continue
            model, tok = load_export(src)
            object.__setattr__(model, "fused_packed_decode", side == "fused")
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            r = measure(
                model,
                tok,
                items[ix],
                spec["max_tokens"],
                budget,
                record_kv_history=True,
            )
            records.append(
                {"sequence": i, "side": side, "item": ix, "repeat": rep, "result": r}
            )
            packed(dest, records)
            print(
                i, side, ix, r["correct"], r["decode_tps"], r["elapsed_s"], flush=True
            )
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(out / "decision.json", outcome(spec, records, items, old))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
