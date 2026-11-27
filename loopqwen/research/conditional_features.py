"""Freeze every causal target position from the native shared instruction backbone."""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..conditional_reentry import PACKED, from_instruction
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..kernels.gdn_backend import select
from ..looped_model import LoopedModel
from ..recurrence import Span
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .native_function_data import token_pair
from .registered import registered
from .table_reentry_fit import root_for

NAME = "conditional-features-01"
CODES = (
    "loopqwen/conditional_reentry.py",
    "loopqwen/research/conditional_features.py",
    "loopqwen/research/native_function_data.py",
    "loopqwen/looped_model.py",
    "loopqwen/tool_model.py",
    "loopqwen/research/projected_distillation.py",
    "scripts/program_language_material.py",
    "uv.lock",
)
CACHE = Path("checkpoints/research")
READING = Path("docs/measurements/table-policy-reading.json")
SOURCE = Path("research/results/table-policy-gradient-01")


def changed_token_spans(left, right):
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
    if start == min(len(left), len(right)):
        raise ValueError(
            "The complete terminally delimited programs need a differing action token"
        )
    end = 0
    while (
        len(left) - end > start
        and len(right) - end > start
        and left[-end - 1] == right[-end - 1]
    ):
        end += 1
    # When the edit inserts a token, include the first common continuation on both sides.
    if start in (len(left) - end, len(right) - end):
        end -= 1
    return list(range(start, len(left) - end)), list(range(start, len(right) - end))


def feature_root(spec):
    return CACHE / spec["id"] / "positions"


def parity(net, prefix):
    x = mx.array([prefix])
    net.eval()
    dense = LoopedModel(net.text, Span(start=20, length=4, passes=1))
    expected = dense.project(dense.hidden(x)[:, -1:])
    direct = net.project(net.last_hidden(x, net.make_cache()))
    cache = net.make_cache()
    cut = len(prefix) // 2
    net.hidden(x[:, :cut], cache)
    states = net.features(x[:, cut:], cache, positions=slice(-1, None))
    incremental = net.project(net.text.model.norm(net.readout(states)))
    source = states[..., 0 if net.span.passes == 2 else -1, :]
    return {
        "vocab": expected.shape[-1],
        "same_ordinary": mx.array_equal(expected, direct).item(),
        "same_initialized_state": mx.array_equal(net.readout(states), source).item(),
        "top_full": mx.argmax(direct).item(),
        "top_chunk": mx.argmax(incremental).item(),
        "max_chunk_logit_difference": mx.max(
            mx.abs(direct.astype(mx.float32) - incremental)
        ).item(),
        "sites": len(cache.slots),
        "independent_slots": len({id(v) for v in cache.slots.values()}),
        "suffix_state_changes": [
            mx.mean(
                mx.abs((states[..., i + 1, :] - states[..., i, :]).astype(mx.float32))
            ).item()
            for i in range(4)
        ],
    }


def one(net, p, y, ix):
    ids = mx.array([p + y[:-1]])
    return net.features(ids, positions=mx.array(ix) + len(p) - 1)[0]


def write_one(tok, net, row, path):
    p, y = token_pair(tok, row)
    data = {"positive": one(net, p, y, list(range(len(y)))), "labels": mx.array(y)}
    negative = []
    for j, n in enumerate(row.get("rejected", [])):
        other, z = token_pair(tok, {**row, "target": n["target"]})
        assert p == other
        left, right = changed_token_spans(y, z)
        data[f"negative_{j}"] = one(net, p, z, right)
        data[f"labels_{j}"] = mx.array([z[i] for i in right])
        negative.append(
            {
                "edit": n["edit"],
                "positive_positions": left,
                "negative_positions": right,
                "negative_labels": fingerprint(z),
            }
        )
    tmp = path.with_suffix(".partial.safetensors")
    mx.save_safetensors(str(tmp), data)
    tmp.replace(path)
    return {
        "id": row["id"],
        "group": row["group"],
        "prefix": fingerprint(p),
        "labels": fingerprint(y),
        "positions": len(y),
        "prefix_tokens": len(p),
        "negative": negative,
        "file": path.name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def qualifies(d, s):
    p = d["parity"]
    return (
        p["same_ordinary"]
        and p["same_initialized_state"]
        and p["top_full"] == p["top_chunk"]
        and p["vocab"] == 248320
        and p["sites"] == p["independent_slots"] == 20 + 4 * s["passes"]
        and min(p["suffix_state_changes"]) > 0
        and d["examples"] == s["examples"]
        and d["positive_positions"] == s["positions"]
        and d["negative_pairs"] == 176
        and d["positive_positions"] >= 500000
    )


def run(name):
    s = registered(name)
    if s["passes"] == 2:
        prior = read(SOURCE / "decision.json")
        if "arms" not in prior or (prior["passed"] and read(READING)["reading_gate"]):
            raise ValueError(
                "The previous primary must be complete and miss its joint gate"
            )
        assert READING.is_file()
    root = root_for(s)
    capture(root, CODES)
    cache = feature_root(s)
    cache.mkdir(exist_ok=True, parents=True)
    guard = Budget(name, s["limits"])
    records = []
    try:
        tok = tokenizer(s)
        net = from_instruction(s)
        select(net, "native")
        rows = unpack(Path(s["training"]))
        chosen = min(
            rows, key=lambda r: fingerprint([s["fit_seed"], "parity", r["id"]])
        )
        p, _ = token_pair(tok, chosen)
        check = parity(net, p)
        save(root / "initial-parity.json", check)
        if not check["same_ordinary"] or check["top_full"] != check["top_chunk"]:
            raise InterruptedError(
                "Native initialized logits or cache parity failed on the registered opening"
            )
        dest = Path(s["checkpoint"]).parent / "conditional-reentry-backbone" / PACKED
        dest.parent.mkdir(exist_ok=True, parents=True)
        tmp = dest.with_suffix(".partial.safetensors")
        mx.save_safetensors(str(tmp), dict(tree_flatten(net.text.parameters())))
        if dest.exists() and digest(tmp) != digest(dest):
            raise ValueError(
                "Existing packed instruction model disagrees with the native conversion"
            )
        tmp.replace(dest)
        identity = {
            "sha256": digest(dest),
            "bytes": dest.stat().st_size,
            "donor": s["donor"]["sha256"],
            "tensors": len(tree_flatten(net.text.parameters())),
        }
        save(root / "backbone.json", identity)
        journal = cache.parent / "partial-index.json.gz"
        if journal.exists():
            records = unpack(journal)
            assert all(
                row["id"] == r["id"] and digest(cache / r["file"]) == r["sha256"]
                for row, r in zip(rows[: len(records)], records, strict=True)
            )
        for i in range(len(records), len(rows)):
            guard.check()
            records.append(write_one(tok, net, rows[i], cache / f"{i:05}.safetensors"))
            if len(records) % 128 == 0:
                packed(journal, records)
                print(
                    "Native feature records", len(records), "of", len(rows), flush=True
                )
        packed(root / "feature-index.json.gz", records)
        d = {
            "parity": check,
            "examples": len(records),
            "positive_positions": sum(r["positions"] for r in records),
            "negative_pairs": sum(len(r["negative"]) for r in records),
            "negative_positions": sum(
                len(n["negative_positions"]) for r in records for n in r["negative"]
            ),
            "feature_bytes": sum(r["bytes"] for r in records),
        }
        d["passed"] = qualifies(d, s)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        packed(root / "partial-index.json.gz", records)
        d = {
            "passed": False,
            "status": "interrupted",
            "reason": str(e),
            "durable_records": len(records),
        }
    finally:
        guard.close()
    save(root / "decision.json", d)


def verify(root, d):
    s = read(root / "registration.json")
    assert s == registered(s["id"])
    if d.get("status") == "interrupted":
        assert (
            not d["passed"]
            and d["reason"]
            and d["durable_records"] == len(unpack(root / "partial-index.json.gz"))
        )
        return
    index = unpack(root / "feature-index.json.gz")
    rows = unpack(Path(s["training"]))
    tok = tokenizer(s)
    assert d["passed"] == qualifies(d, s) and len(index) == len(rows) == d["examples"]
    assert d["parity"] == read(root / "initial-parity.json")
    assert sum(r["bytes"] for r in index) == d["feature_bytes"]
    assert sum(r["positions"] for r in index) == d["positive_positions"]
    assert sum(len(r["negative"]) for r in index) == d["negative_pairs"]
    assert (
        sum(len(n["negative_positions"]) for r in index for n in r["negative"])
        == d["negative_positions"]
    )
    for i, (r, row) in enumerate(zip(index, rows, strict=True)):
        p, y = token_pair(tok, row)
        assert (
            r["id"] == row["id"]
            and r["group"] == row["group"]
            and r["prefix"] == fingerprint(p)
            and r["labels"] == fingerprint(y)
            and r["positions"] == len(y)
            and r["prefix_tokens"] == len(p)
        )
        assert r["file"] == f"{i:05}.safetensors" and len(r["sha256"]) == 64
        for n, other in zip(r["negative"], row.get("rejected", []), strict=True):
            _, z = token_pair(tok, {**row, "target": other["target"]})
            assert n["negative_labels"] == fingerprint(z) and changed_token_spans(
                y, z
            ) == (n["positive_positions"], n["negative_positions"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=NAME)
    run(parser.parse_args().name)
