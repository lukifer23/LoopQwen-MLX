"""Measure what the learned source query actually identifies at each supervised decision."""

import argparse
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import read, digest
from loopqwen.research.conditional_features import CACHE
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.register_bounded_read import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/bounded-read-source-address.json")
OUT = Path("docs/measurements/bounded-read-source-address")


def run(freeze):
    root = RESULT / FIT
    for p in (
        root / "checksums.json",
        root / "feature-index.json.gz",
        root / "decision.json",
    ):
        committed(p)
    cfg = read(root / "registration.json")
    assert read(root / "decision.json")["passed"]
    specs = dict(
        fit=digest(root / "checksums.json"),
        software=digest(Path(__file__)),
        material=digest(Path(cfg["training"])),
        protocol="Descriptive mechanism audit on all 1,352 location-supervised training decisions, across 320 HotpotQA comparisons and 320 arithmetic calls, with the full original opening and causal teacher-forced output prefix. Use the frozen bounded model. At each designated target position compare the final query after one and two real visits at the same weights, averaging normalized attention probabilities over the two content heads. Record actual cited-span mass and top-position hits; source-only probability of the exact next token ID and its top-token hit when pooling repeated opening IDs; fraction of content attention on the original opening; native vocabulary target probability and top1. Compare a uniform opening copy distribution to expose frequency. Repeated IDs can match an incorrect field and tokenization can differ at boundaries. No trained copy mechanism or generated answer is scored. Require all original feature identities and all registered decisions, native finite probability and consistent shapes; stop at 240 seconds, 3 GiB or OS memory pressure, without testing a held answer. An exact pointer intervention is worth testing only if two reads give a true-target source-only probability above the uniform opening probability on >50% of financial decisions and a greater top-token hit count than uniform in that group.",
        limits=dict(seconds=240, peak_gib=3, available_gib=1.5),
    )
    if freeze:
        persist(PLAN, specs)
        return
    committed(PLAN)
    assert specs == read(PLAN) and not (OUT / "summary.json").exists()
    start = monotonic()
    next_check = 0

    def guard():
        nonlocal next_check
        now = monotonic()
        if now < next_check:
            return
        next_check = now + 0.8
        if (
            now - start > 240
            or mx.get_peak_memory() > 3 * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("native source attention budget")

    mx.reset_peak_memory()
    actor = SharedQueryAssistant(cfg["checkpoint"], root)
    net = actor._tool_model
    features = RESULT / cfg["feature_source"]
    folder = CACHE / cfg["feature_source"] / digest(features / "registration.json")
    source = unpack(Path(cfg["training"]))
    index = unpack(root / "feature-index.json.gz")
    records = []
    stop = None
    try:
        for w, r in zip(source, index, strict=True):
            if not w["annotations"]:
                continue
            guard()
            path = folder / r["file"]
            assert w["id"] == r["id"] and digest(path) == r["sha256"]
            z = mx.load(str(path))
            p = w["p"]
            source_ids = mx.array(p)
            from collections import Counter

            freq = Counter(p)
            maximum = max(freq.values())
            for mark in w["annotations"]:
                j = mark["query"]
                y = w["y"][j]
                assert z["labels"][j].item() == y
                pos = len(p) - 1 + j
                k = z["keys"][:, :, : pos + 1]
                v = z["values"][:, :, : pos + 1]
                row = dict(
                    id=w["id"],
                    group=w["group"],
                    query=j,
                    target=y,
                    opening=len(p),
                    matched_opening=freq[y],
                    chance=freq[y] / len(p),
                    uniform_top=int(
                        y == min(k for k, n in freq.items() if n == maximum)
                    ),
                    ranges=mark["source"],
                    paths={},
                )
                for visits in (1, 2):
                    q = []
                    state = net.refine(
                        z["pre"][:, j : j + 1],
                        z["ordinary"][:, j : j + 1],
                        k,
                        v,
                        pos,
                        None,
                        visits=visits,
                        observe_content=q.append,
                    )
                    qq = q[0].astype(mx.float32)
                    weights = mx.softmax(
                        (qq * net.step.dimension**-0.5)
                        @ v.astype(mx.float32).swapaxes(-1, -2),
                        axis=-1,
                    ).mean(axis=1)[0, 0]
                    limit = weights[: len(p)]
                    tokens = mx.zeros((net.args.vocab_size,)).at[source_ids].add(limit)
                    predicted = mx.softmax(
                        net.project(net.text.model.norm(state))[0, 0].astype(mx.float32)
                    )
                    peak = mx.argmax(weights).item()
                    assert abs(weights.sum().item() - 1) < 0.0001
                    mass = sum(weights[lo:hi].sum().item() for lo, hi in mark["source"])
                    row["paths"][str(visits)] = dict(
                        span_mass=mass,
                        source_hit=any(a <= peak < b for a, b in mark["source"]),
                        top_position=peak,
                        opening_mass=limit.sum().item(),
                        copy_probability=tokens[y].item(),
                        copy_hit=mx.argmax(tokens).item() == y,
                        base_probability=predicted[y].item(),
                        native_hit=mx.argmax(predicted).item() == y,
                    )
                records.append(row)
        guard()
    except InterruptedError as err:
        stop = str(err)
    sums = defaultdict(lambda: defaultdict(float))
    for r in records:
        for v, path in r["paths"].items():
            total = sums[r["group"] + "/" + v]
            total["n"] += 1
            for key in (
                "span_mass",
                "source_hit",
                "opening_mass",
                "copy_probability",
                "copy_hit",
                "base_probability",
                "native_hit",
            ):
                total[key] += path[key]
            total["uniform_top"] += r["uniform_top"]
            total["better_than_uniform"] += (
                path["copy_probability"] > r["chance"] + 1e-8
            )
    verified = (
        len(records) == 1352
        and all(math.isfinite(x) for group in sums.values() for x in group.values())
        and stop is None
    )
    cal = sums.get("calculate/2", {})
    packed(OUT / "decisions.json.gz", records)
    persist(
        OUT / "summary.json",
        dict(
            passed=verified,
            stop=stop,
            positions=len(records),
            sums=sums,
            plan=digest(PLAN),
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
            data=digest(OUT / "decisions.json.gz"),
            copy_investigation=bool(
                verified
                and cal["better_than_uniform"] > cal["n"] / 2
                and cal["copy_hit"] > cal["uniform_top"]
            ),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
