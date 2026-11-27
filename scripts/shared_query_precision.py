"""Audit precision conversion of the packed donor and its actual cached outputs."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from loopqwen.conditional_reentry import PACKED
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_features import FEATURES, OLD, RESULTS
from loopqwen.shared_query_assistant import from_backbone
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/shared-query-precision.json")
REPORT = Path("docs/measurements/shared-query-precision.json")
INVENTORY = Path("docs/measurements/shared-query-precision-storage.json")


def storage():
    ident = read(OLD / "backbone.json")
    path = Path("exports/conditional-reentry-backbone") / PACKED
    assert digest(path) == ident["sha256"]
    with path.open("rb") as f:
        length = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(length))
    values = np.memmap(path, mode="r", dtype=np.uint8, offset=length + 8)
    c = Counter()
    changed = {}
    max_error = 0.0
    for key, r in header.items():
        if key == "__metadata__":
            continue
        c[r["dtype"] + "_tensors"] += 1
        if r["dtype"] != "BF16":
            continue
        a, b = r["data_offsets"]
        raw = values[a:b].view("<u2")
        for i in range(0, len(raw), 262144):
            source = (raw[i : i + 262144].astype("uint32") << 16).view("float32")
            with np.errstate(under="ignore", over="ignore"):
                cast = source.astype("float16").astype("float32")
            assert np.isfinite(cast).all()
            count = int((cast != source).sum())
            if count:
                changed[key] = changed.get(key, 0) + count
                max_error = max(max_error, float(abs(source - cast).max()))
            c["BF16_scalars"] += source.size
            c["inexact_round_trips"] += count
    receipt = {
        "input": ident,
        "counts": dict(c),
        "changed_per_tensor": changed,
        "max_abs_error": max_error,
        "conversion": "Keep packed uint32 weights and all native float32 parameters unchanged. Cast only the original bfloat16 parameters to float16, preserving 16-bit storage.",
    }
    persist(INVENTORY, receipt)
    return receipt


def cases(s):
    rows = unpack(Path(s["training"]))
    selected = []
    for g in ("strategy", "report_action", "instruction"):
        pool = [r for r in rows if r["group"] == g]
        original = min(pool, key=lambda w: fingerprint([s["fit_seed"], w["id"]]))
        more = sorted(
            (r for r in pool if r is not original),
            key=lambda w: fingerprint([94137, w["id"]]),
        )[:3]
        selected += [original, *more]
    return selected


def run():
    committed(PLAN)
    committed(INVENTORY)
    s = read(RESULTS / FEATURES / "registration.json")
    cfg = read(PLAN)
    assert digest(INVENTORY) == cfg["storage"]
    chosen = cases(s)
    assert [r["id"] for r in chosen] == cfg["ids"]
    tok = tokenizer(s)
    start = perf_counter()

    def guard():
        import subprocess

        import psutil

        if (
            perf_counter() - start > cfg["max_seconds"]
            or mx.get_peak_memory() > cfg["max_mlx_bytes"]
        ):
            raise InterruptedError("Precision diagnostic resource cap")
        if (
            psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("System memory warning")

    net = from_backbone(
        s["checkpoint"], RESULTS / FEATURES, s, read(OLD / "backbone.json")
    )
    stock = LoopedModel(net.text, Span(len(net.text.layers) - 1, 1, 1))
    stock.eval()
    inputs = [mx.array([token_pair(tok, w)[0]]) for w in chosen]
    originals = []
    records = []
    try:
        for ids in inputs:
            guard()
            out = stock.project(stock.hidden(ids, stock.make_cache())[:, -1:]).astype(
                mx.float32
            )
            mx.eval(out)
            originals.append(out)
        net.text.set_dtype(mx.float16, lambda dtype: dtype == mx.bfloat16)
        mx.eval(net.text.parameters())
        for w, ids, old in zip(chosen, inputs, originals, strict=True):
            guard()
            results = []
            slots = None
            for actor in (stock, net):
                for split in (False, True):
                    c = actor.make_cache()
                    mid = ids.shape[1] // 2 if split else 0
                    if mid:
                        actor.hidden(ids[:, :mid], c)
                    h = (
                        actor.hidden(ids[:, mid:], c)[:, -1:]
                        if actor is stock
                        else actor.last_hidden(ids[:, mid:], c)
                    )
                    out = actor.project(h).astype(mx.float32)
                    mx.eval(out)
                    results.append(out)
                    slots = [
                        int(slot.offset)
                        for (i, _), slot in c.slots.items()
                        if not net.text.layers[i].is_linear
                    ]
                    assert len(c.slots) == 24 and slots == [ids.shape[1]] * 6

            def dist(a, b):
                return mx.max(mx.abs(a - b)).item()

            a, b, c, d = results
            tops = [mx.argmax(x).item() for x in (old, a, b, c, d)]
            records.append(
                {
                    "id": w["id"],
                    "tokens": ids.shape[1],
                    "ordinary": dist(a, b),
                    "recurrent": dist(c, d),
                    "architectures_full": dist(a, c),
                    "architectures_split": dist(b, d),
                    "bf16_to_fp16": dist(old, a),
                    "top": tops,
                    "native_seconds": perf_counter() - start,
                    "offsets": slots,
                }
            )
        same = sum(len(set(r["top"])) == 1 for r in records)
        passed = (
            all(
                max(
                    r[k]
                    for k in (
                        "ordinary",
                        "recurrent",
                        "architectures_full",
                        "architectures_split",
                    )
                )
                <= 0.55
                and len(set(r["top"][1:])) == 1
                for r in records
            )
            and same >= 11
            and all(len(set(records[i]["top"])) == 1 for i in (0, 4, 8))
        )
        status = "completed"
    except (InterruptedError, TimeoutError, MemoryError) as e:
        status = str(e)
        passed = False
    result = {
        "plan": digest(PLAN),
        "passed": passed,
        "status": status,
        "records": records,
        "seconds": perf_counter() - start,
        "peak_bytes": mx.get_peak_memory(),
    }
    persist(REPORT, result)
    print(result["status"], result["passed"], len(records))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "measure"))
    a = p.parse_args()
    if a.mode == "measure":
        run()
    else:
        rc = storage()
        s = read(RESULTS / FEATURES / "registration.json")
        persist(
            PLAN,
            {
                "storage": digest(INVENTORY),
                "failed": digest(RESULTS / FEATURES / "checksums.json"),
                "matched_diagnostic": digest(
                    Path("docs/measurements/shared-query-partition.json")
                ),
                "question": "Can FP16 arithmetic preserve the original cache tolerance on the deployed packed Qwen without widening its 16-bit weights or modifying the original 4-bit matrix codes?",
                "ids": [r["id"] for r in cases(s)],
                "max_seconds": 180,
                "max_mlx_bytes": 2147483648,
                "threshold": "All 12 hash-selected training prompts must reproduce the top token and maximum 0.55 logit error across both cache partitions and both architectures in FP16. Preserve the previous BF16 ordinary top token on all three earlier failed prompts and at least 11 of 12 overall. Stop at either resource bound or a macOS memory warning. Only a numerically successful method can be enrolled for source-repair fitting. No held prompts, new generation arms or task claims.",
            },
        )
        print(rc["counts"])
