"""Map-disjoint RSI pools and deterministic coverage-matched curricula."""

import collections
import hashlib
import json
import os
import random
from pathlib import Path

from ..identity import fingerprint
from ..io import read
from . import pointer5

REG = Path(
    "experiments/rsi_replication.json"
    if os.environ.get("LOOPQWEN_RSI_REPLICATION") == "1"
    else "experiments/rsi_pilot.json"
)
S = read(REG)


def historical():
    used = set()

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if (
                    k in ("map", "mapping")
                    and isinstance(v, list)
                    and sorted(v) == list(range(8))
                ):
                    used.add(hashlib.sha256(bytes(v)).hexdigest())
                elif k in (
                    "transition",
                    "original",
                    "cross",
                    "relabel",
                    "off_path",
                    "on_path",
                    "other_start",
                    "same_state",
                    "order",
                    "calibration",
                    "comparison",
                ):
                    visit(v)
                elif isinstance(v, dict):
                    visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for pattern in ("phase5_pointer/*.jsonl", "v4_pointer/*.jsonl", "v4/*.jsonl"):
        for p in sorted(Path("data").glob(pattern)):
            with p.open() as f:
                for line in f:
                    visit(json.loads(line))
    visit(read("docs/measurements/phase7/language/cohorts.json"))
    return used


def _pools(spec, used):
    initial = fingerprint(sorted(used))
    rng = random.Random(spec["seed"])
    splits = {}
    for name, size in [
        ("development", spec["development_maps"]),
        ("confirmation", spec["confirmation_maps"]),
        *(
            (f"pool{i + 1}", spec["candidate_maps_per_round"])
            for i in range(spec["rounds"])
        ),
    ]:
        rows = []
        counts = collections.Counter()
        while len(rows) < size:
            mapping = rng.sample(range(8), 8)
            start = rng.randrange(8)
            r = pointer5.instance(mapping, start)
            cycle = pointer5.cycle(r)
            key = (start, cycle >= 6)
            if r["map_id"] in used or cycle < 4 or counts[key] >= size // 16:
                continue
            used.add(r["map_id"])
            counts[key] += 1
            rows.append(
                {
                    **r,
                    "cycle": cycle,
                    "stratum": f"{start}/{'long' if cycle >= 6 else 'short'}",
                }
            )
        splits[name] = rows
    return splits, initial


def pools():
    used = historical()
    if S["id"] == "rsi2":
        old, _ = _pools(read("experiments/rsi_pilot.json"), used.copy())
        used.update(r["map_id"] for rows in old.values() for r in rows)
        trace = read("docs/measurements/rsi_trace1/splits.json")
        used.update(r["map_id"] for rows in trace.values() for r in rows)
    return _pools(S, used)


def select(rows, diag, round_id):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["stratum"]].append(r)
    selected = []
    for items in groups.values():
        if diag is None:
            chosen = [
                (r, "static")
                for r in sorted(
                    items,
                    key=lambda q: fingerprint(
                        ["static", S["seed"], round_id, q["map_id"]]
                    ),
                )[:8]
            ]
        else:
            buckets = {k: [] for k in ("boundary", "failure", "mastered")}
            for r in items:
                a, b = diag[r["map_id"]]
                prefix = max(a["prefix"], b["prefix"])
                if min(a["prefix"], b["prefix"]) == 6:
                    group = "mastered"
                elif (
                    a["pred"] != b["pred"]
                    or 4 <= a["prefix"] <= 5
                    or 4 <= b["prefix"] <= 5
                ):
                    group = "boundary"
                else:
                    group = "failure"
                conf = (a["true_p"][-1] + b["true_p"][-1]) / 2
                rank = {
                    "mastered": conf,
                    "failure": -prefix,
                    "boundary": abs(conf - 0.5),
                }[group]
                buckets[group].append((rank, r["map_id"], r))
            chosen = []
            extra = []
            for group, quota in zip(buckets, (5, 2, 1), strict=True):
                order = sorted(buckets[group])
                chosen.extend((r, group) for _, _, r in order[:quota])
                extra.extend((r, group + "-fill") for _, _, r in order[quota:])
            chosen += extra[: 8 - len(chosen)]
        selected += chosen
    selected.sort(key=lambda pair: fingerprint(["batch", round_id, pair[0]["map_id"]]))
    if len(selected) != S["selected_maps_per_arm_round"]:
        raise ValueError("Unbalanced curriculum")
    return selected
