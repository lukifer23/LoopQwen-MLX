"""Evaluate contiguous full-history KV reservation on the qualified language decoder."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import interpret
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from .balanced_register import flatten
from .core import Budget
from .ledger_curriculum import measure
from .registered import registered


def outcome(spec, rows, items):
    costs = {
        name: dict(
            ttft=sum(r["ttft_s"] for r in rs),
            seconds=sum(r["elapsed_s"] for r in rs),
            peak=max(r["peak_bytes"] for r in rs),
            correct=sum(r["correct"] for r in rs),
        )
        for name, rs in rows.items()
    }
    c, b = costs["reserved"], costs["stock"]
    exact = True
    for a, o, w in zip(rows["reserved"], rows["stock"], items, strict=True):
        sec = a["attention_memory"]
        sites = a["reserved_sites"]
        exact &= (
            a["input_sha256"] == o["input_sha256"] == fingerprint(w["prompt_ids"])
            and a["token_ids"] == o["token_ids"]
            and sec == o["attention_memory"]
            and sec["start"] <= w["gold_start"] <= w["gold_stop"] <= sec["stop"]
            and a["context_route"] == o["context_route"] == "register"
            and {(s["layer"], s["turn"]) for s in sites}
            == {(3, 0), (7, 0), (11, 0), (11, 1), (15, 0), (15, 1), (19, 0), (23, 0)}
            and all(
                s["capacity"] == len(w["prompt_ids"]) + spec["max_tokens"] - 1
                and s["offset"] == len(w["prompt_ids"]) + len(a["token_ids"]) - 1
                for s in sites
            )
        )
    return dict(
        passed=bool(
            exact
            and c["correct"] == len(items)
            and c["ttft"] <= spec["ratio"] * b["ttft"]
            and c["seconds"] <= b["seconds"]
            and c["peak"] <= b["peak"]
        ),
        exact=exact,
        metrics=costs,
        items=len(items),
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    items = flatten(spec)
    tok = load_tokenizer(local_snapshot())
    term = {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
    rows = {k: unpack(root / (k + ".json.gz")) for k in ("reserved", "stock")}
    for rs in rows.values():
        for row, w in zip(rs, items, strict=True):
            ids = row["token_ids"]
            parsed = interpret(
                ids,
                tok.convert_tokens_to_ids("<think>"),
                tok.convert_tokens_to_ids("</think>"),
                term,
                tok,
            )
            answer = numeric(parsed["visible"]) if ids[-1] in term else None
            assert answer == row["answer"] and (answer == w["gold"]) == row["correct"]
            assert row["input_sha256"] == fingerprint(w["prompt_ids"]) and row[
                "prompt_tokens"
            ] == len(w["prompt_ids"])
    assert outcome(spec, rows, items) == decision


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    parent = Path(spec["requires"])
    assert (
        digest(parent / "checksums.json") == spec["prerequisite_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    budget = Budget(name, spec["limits"])
    rows = {"reserved": [], "stock": []}
    items = flatten(spec)
    try:
        for i, item in enumerate(items):
            for side in ("reserved", "stock") if i % 2 == 0 else ("stock", "reserved"):
                model, tok, _ = load_trained(spec["checkpoint"])
                mx.reset_peak_memory()
                record = measure(
                    model,
                    tok,
                    item,
                    spec["max_tokens"],
                    budget,
                    reserve_full_kv=side == "reserved",
                )
                rows[side].append(record)
                print(
                    i,
                    side,
                    record["correct"],
                    record["ttft_s"],
                    record["peak_bytes"],
                    flush=True,
                )
                del model, tok
                gc.collect()
                mx.clear_cache()
        for side, rs in rows.items():
            packed(root / (side + ".json.gz"), rs)
        save(root / "decision.json", outcome(spec, rows, items))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
