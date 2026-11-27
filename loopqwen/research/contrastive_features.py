"""Extract the causal states for every qualified human pair under a frozen physical second visit."""

import argparse
from pathlib import Path

import mlx.core as mx

from ..conditional_reentry import resident
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..kernels.gdn_backend import select
from .conditional_features import (
    CODES,
    changed_token_spans,
    feature_root,
    one,
    parity,
)
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .native_function_data import token_pair
from .registered import registered
from .table_reentry_fit import root_for

SOURCES = (
    *CODES,
    "loopqwen/contrastive_reentry.py",
    "loopqwen/research/contrastive_features.py",
    "scripts/contrastive_world_material.py",
)
FEATURES = ("contrastive-table-features-01", "contrastive-language-features-01")
FIT = "contrastive-world-fit-01"
OLD = Path("research/results/conditional-features-01")
PREVIOUS = Path("research/results/conditional-program-primary-01")


def record(net, tok, row, path):
    p, y = token_pair(tok, row)
    allpos = one(net, p, y, list(range(len(y))))
    arrays = {"positive": allpos, "labels": mx.array(y)}
    alternatives = []
    for j, other in enumerate(row.get("rejected", [])):
        opening, z = token_pair(tok, {**row, "target": other["target"]})
        assert p == opening
        good, bad = changed_token_spans(y, z)
        assert good[0] == bad[0] and y[: good[0]] == z[: bad[0]]
        first = bad[0]
        # Both first differing predictions condition on exactly the same prefix. A one-token
        # negative needs only its different label; retain the original physical states.
        if len(bad) == 1:
            neg = allpos[first : first + 1]
        else:
            neg = one(net, p, z, bad)
        arrays[f"negative_{j}"] = neg
        arrays[f"labels_{j}"] = mx.array([z[t] for t in bad])
        alternatives.append(
            {
                "edit": other["edit"],
                "positive_positions": good,
                "negative_positions": bad,
                "negative_labels": fingerprint(z),
                "same_prefix_only": len(bad) == 1,
            }
        )
    tmp = path.with_suffix(".partial.safetensors")
    mx.save_safetensors(str(tmp), arrays)
    tmp.replace(path)
    return {
        "id": row["id"],
        "group": row["group"],
        "prefix": fingerprint(p),
        "labels": fingerprint(y),
        "positions": len(y),
        "prefix_tokens": len(p),
        "negative": alternatives,
        "file": path.name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def cohort(s):
    return [r for r in unpack(Path(s["training"])) if r["group"] in s["groups"]]


def report(idx, check, s):
    summary = {
        "examples": len(idx),
        "positions": sum(r["positions"] for r in idx),
        "alternatives": sum(len(r["negative"]) for r in idx),
        "negative_positions": sum(
            len(z["negative_positions"]) for r in idx for z in r["negative"]
        ),
        "same_causal_prefix_reuses": sum(
            z["same_prefix_only"] for r in idx for z in r["negative"]
        ),
        "feature_bytes": sum(r["bytes"] for r in idx),
    }
    par = (
        check["same_ordinary"]
        and check["cached_same_ordinary"]
        and check["same_initialized_state"]
        and check["top_chunk"] == check["top_full"]
        and check["sites"] == check["independent_slots"] == 20 + 4 * s["passes"]
        and min(check["suffix_state_changes"]) > 0
    )
    return {
        **summary,
        "parity_passed": par,
        "passed": bool(par and all(summary[k] == v for k, v in s["expected"].items())),
    }


def verify(root, d):
    s = read(root / "registration.json")
    assert s == registered(s["id"]) and read(root / "backbone.json") == read(
        OLD / "backbone.json"
    )
    if d.get("status") == "interrupted":
        assert (
            not d["passed"]
            and d["reason"]
            and d["durable_records"] == len(unpack(root / "partial-index.json.gz"))
        )
        return
    idx = unpack(root / "feature-index.json.gz")
    assert report(idx, read(root / "initial-parity.json"), s) == d
    tok = tokenizer(s)
    for i, (row, r) in enumerate(zip(cohort(s), idx, strict=True)):
        p, y = token_pair(tok, row)
        assert (
            r["id"] == row["id"]
            and r["group"] == row["group"]
            and r["file"] == f"{i:05}.safetensors"
            and len(r["sha256"]) == 64
        )
        assert (
            r["prefix"] == fingerprint(p)
            and r["labels"] == fingerprint(y)
            and r["positions"] == len(y)
            and r["prefix_tokens"] == len(p)
        )
        for z, bad in zip(r["negative"], row.get("rejected", []), strict=True):
            op, labels = token_pair(tok, {**row, "target": bad["target"]})
            a, b = changed_token_spans(y, labels)
            assert op == p and z == {
                "edit": bad["edit"],
                "positive_positions": a,
                "negative_positions": b,
                "negative_labels": fingerprint(labels),
                "same_prefix_only": len(b) == 1,
            }


def run(name):
    s = registered(name)
    scoring = read("docs/measurements/conditional-program-reading.json")
    if read(PREVIOUS / "decision.json")["passed"] and scoring["passed"]:
        raise ValueError("The preceding checkpoint qualified for replication")
    root = root_for(s)
    capture(root, SOURCES)
    ident = read(OLD / "backbone.json")
    save(root / "backbone.json", ident)
    guard = Budget(name, s["limits"])
    where = feature_root(s)
    where.mkdir(exist_ok=True, parents=True)
    journal = where.parent / "partial-index.json.gz"
    idx = unpack(journal) if journal.exists() else []
    try:
        tok = tokenizer(s)
        net = resident(s["checkpoint"], root, s, ident)
        select(net, "native")
        rows = cohort(s)
        assert all(
            r["id"] == w["id"] and digest(where / r["file"]) == r["sha256"]
            for r, w in zip(idx, rows[: len(idx)], strict=True)
        )
        probe = min(rows, key=lambda w: fingerprint([s["fit_seed"], name, w["id"]]))
        p, _ = token_pair(tok, probe)
        check = parity(net, p)
        save(root / "initial-parity.json", check)
        if not report([], check, s)["parity_passed"]:
            raise InterruptedError(
                "Initial native parity or physical-site checks failed"
            )
        for i in range(len(idx), len(rows)):
            guard.check()
            idx.append(record(net, tok, rows[i], where / f"{i:05}.safetensors"))
            if len(idx) % 128 == 0:
                packed(journal, idx)
                print("Native", len(idx), "of", len(rows), flush=True)
        packed(root / "feature-index.json.gz", idx)
        d = report(idx, check, s)
    except (InterruptedError, TimeoutError, MemoryError) as e:
        packed(root / "partial-index.json.gz", idx)
        d = {
            "status": "interrupted",
            "passed": False,
            "reason": str(e),
            "durable_records": len(idx),
        }
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=FEATURES)
    run(parser.parse_args().study)
