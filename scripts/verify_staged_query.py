"""Compare full-vocabulary logits in both storage modes and exercise a real full-length source."""

import argparse
import gc
import gzip
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen.evaluation.generation_eval import advance_cache
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.offloaded_prefill import OffloadedPrefill, PrefixOnDisk
from loopqwen.ordinary_model import ordinary
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.locked_context_material import SOURCE, VERSIONS, blob
from scripts.register_repair_query import FIT, PRIMARY, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/staged-query-memory.json")
REPORT = Path("docs/measurements/staged-query-memory.json")


def material():
    spec = read(RESULT / FIT / "registration.json")
    tok = tokenizer(spec)
    rows = unpack(Path(spec["training"]))
    inputs = {}
    for group in ("strategy", "report_action", "redirect_valid_read"):
        for r in sorted(
            (r for r in rows if r["group"] == group),
            key=lambda x: fingerprint(["stage parity", x["id"]]),
        ):
            ids, _ = token_pair(tok, r)
            if 257 <= len(ids) <= 1536:
                inputs[r["id"]] = ids
                break
    assert len(inputs) == 3
    ref, expected = VERSIONS[0]
    file = SOURCE / (ref + ".lock.gz")
    original = gzip.decompress(file.read_bytes())
    assert blob(original) == expected
    user = (
        "Here is the complete uv.lock.\n\n"
        + original.decode()
        + '\nEnd of file.\nWhich exact version of the package "virtue" is pinned?'
    )
    ids = tok.encode(
        render(tok, [{"role": "user", "content": user}], []), add_special_tokens=False
    )
    assert len(ids) > 131072
    plan = {
        "fit": digest(RESULT / FIT / "checksums.json"),
        "interrupted": digest(RESULT / PRIMARY / "checksums.json"),
        "implementation": {
            k: digest(Path(k))
            for k in (
                "loopqwen/offloaded_prefill.py",
                "loopqwen/shared_query.py",
                "loopqwen/evaluation/generation_eval.py",
                "scripts/verify_staged_query.py",
            )
        },
        "short": {
            k: {"tokens": len(v), "ids": fingerprint(v)} for k, v in inputs.items()
        },
        "long": {
            "file": str(file),
            "sha256": digest(file),
            "tokens": len(ids),
            "ids": fingerprint(ids),
        },
        "protocol": "Compare four complete vocabulary outputs per original prompt on each of the trained loop and untouched ordinary Base, with 128-token blocks and identical greedy steps. Require identical token decisions and maximum absolute logit difference at most 0.03125. Require every full attention offset and physical row plus two actual learned reads per emitted loop token. Then process one complete independent archival lockfile on the trained model with four output positions and six full attention histories. Count capacity and actual costs. This engineering run measures numerical fidelity and context processing. Primary accuracy, matched depth and user-facing costs retain their separate registrations.",
        "limits": {
            "native_seconds": 780,
            "peak_gib": 3.15,
            "available_gib": 1.5,
            "scratch_gib": 4,
        },
    }
    return spec, inputs, ids, plan


def resident(net, cache, loop, x):
    h = net.last_hidden(x, cache) if loop else net.hidden(x, cache)[:, -1:]
    return net.project(h)


def measure(freeze):
    spec, inputs, long, plan = material()
    if freeze:
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    limits = plan["limits"]
    start = monotonic()
    result = {"plan": digest(PLAN), "short": [], "long": {}, "passed": False}
    phase = {}
    latest = 0

    def guard(*_):
        nonlocal latest
        if (
            monotonic() - start > limits["native_seconds"]
            or mx.get_peak_memory() > limits["peak_gib"] * 2**30
        ):
            raise InterruptedError("Engineering resource limit")
        if monotonic() - latest > 0.75:
            latest = monotonic()
            mem = psutil.virtual_memory().available / 2**30
            level = subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            if level != "1" or mem < limits["available_gib"]:
                raise InterruptedError(
                    f"System pressure {level}, available {mem:.2f} GiB"
                )

    def check_slots(disk, n):
        vals = [
            (s.offset, s.physical_length())
            for s in disk.cache.slots.values()
            if isinstance(s, PrefixOnDisk)
        ]
        assert len(vals) == 6 and all(x == (n, n) for x in vals)
        return vals

    mx.reset_peak_memory()
    try:
        with TemporaryDirectory() as temp:
            assert shutil.disk_usage(temp).free > limits["scratch_gib"] * 2**30
            for loop in (False, True):
                arm = "loop" if loop else "base"
                phase = {"arm": arm, "phase": "load"}
                net = (
                    SharedQueryAssistant(spec["checkpoint"], RESULT / FIT)._tool_model
                    if loop
                    else ordinary(spec["checkpoint"], require_release=True)[0]
                )
                net.eval()
                for k, ids in inputs.items():
                    phase = {"arm": arm, "id": k, "phase": "parity"}
                    guard()
                    cache = net.make_cache()
                    i = 0
                    while len(ids) - i > 128:
                        advance_cache(net, mx.array([ids[i : i + 128]]), cache)
                        i += 128

                    baseline = resident(net, cache, loop, mx.array([ids[i:]]))
                    with TemporaryDirectory(dir=temp) as tmp:
                        disk = OffloadedPrefill(net, tmp, group_attention=True)
                        before = net.output_revisits if loop else 0
                        actual = disk.prefill(ids, guard)
                        errors = []
                        tokens = []
                        resident_tokens = []
                        for j in range(4):
                            guard()
                            check_slots(disk, len(ids) + j)
                            token = mx.argmax(actual[:, -1], -1).item()
                            t = mx.argmax(baseline[:, -1], -1).item()
                            errors.append(
                                mx.max(
                                    mx.abs(
                                        actual.astype(mx.float32)
                                        - baseline.astype(mx.float32)
                                    )
                                ).item()
                            )
                            tokens.append(token)
                            resident_tokens.append(t)
                            if j < 3:
                                q = mx.array([[t]])
                                actual = disk.decode(q)
                                baseline = resident(net, cache, loop, q)
                        record = {
                            "arm": arm,
                            "id": k,
                            "errors": errors,
                            "staged_tokens": tokens,
                            "resident_tokens": resident_tokens,
                            "attention_sites": 6,
                        }
                        result["short"].append(record)
                        if (
                            tokens != resident_tokens
                            or max(errors) > 0.03125
                            or (loop and net.output_revisits - before != 14)
                        ):
                            raise InterruptedError(
                                "Registered logit, token or read-count prerequisite"
                            )
                    del disk, cache, baseline, actual
                if loop:
                    phase = {"arm": arm, "phase": "long"}
                    with TemporaryDirectory(dir=temp) as tmp:
                        disk = OffloadedPrefill(net, tmp, group_attention=True)

                        def track(*_, phase=phase, disk=disk):
                            phase.update(
                                sites=disk.active_sites, consumed=disk.consumed_in_group
                            )
                            guard()

                        begin = monotonic()
                        n0 = net.output_revisits
                        value = disk.prefill(long, track)
                        tokens = []
                        first = None
                        for j in range(4):
                            guard()
                            t = mx.argmax(value[:, -1], -1).item()
                            tokens.append(t)
                            if first is None:
                                first = monotonic() - begin
                            slots = check_slots(disk, len(long) + j)
                            if j < 3:
                                value = disk.decode(mx.array([[t]]))
                        result["long"] = {
                            "prompt": len(long),
                            "output": tokens,
                            "read_only_visits": net.output_revisits - n0,
                            "histories": slots,
                            "scratch_bytes": disk.scratch_bytes,
                            "first_s": first,
                            "wall_s": monotonic() - begin,
                        }
                        if (
                            net.output_revisits - n0 != 8
                            or disk.scratch_bytes > limits["scratch_gib"] * 2**30
                        ):
                            raise InterruptedError("Long-context physical prerequisite")
                del net
                gc.collect()
                mx.clear_cache()
        result["passed"] = True
    except (MemoryError, InterruptedError) as e:
        result.update(reason=str(e), last_phase=phase)
    result.update(native_seconds=monotonic() - start, peak_bytes=mx.get_peak_memory())
    persist(REPORT, result)
    print(result["passed"], result.get("reason", ""), len(result["short"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "measure"))
    measure(p.parse_args().mode == "freeze")
