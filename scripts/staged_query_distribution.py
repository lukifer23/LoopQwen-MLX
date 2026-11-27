"""Summarize paired user-facing latency and accuracy across independent sources after masking."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from scripts.assistance_deliberation import write_csv
from scripts.register_repair_query import PANEL
from scripts.register_staged_query import STAGE
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/staged-query-distribution.json")
DOC = Path("docs/measurements")
REG = Path("research/experiments") / (STAGE + ".json")
DEADLINES = (1, 2, 5, 10, 30, 60, 120, 240, 480)
METRICS = ("correct", "completed", "wall_s", "first_s")


def run(freeze):
    panel = unpack(PANEL)["rows"]
    families = defaultdict(list)
    for i, w in enumerate(panel):
        families[w["family"]].append(i)
    plan = {
        "registration": digest(REG),
        "panel": digest(PANEL),
        "cluster_sizes": {
            f: dict(Counter(panel[i]["cluster"] for i in ix))
            for f, ix in families.items()
        },
        "seed": 20261127,
        "draws": 10000,
        "deadlines_s": list(DEADLINES),
        "scope": "Secondary descriptive extension for the original complete pair. After source judgments are sealed, report raw p50/p95 elapsed and first-output seconds for each arm, every family's complete and correct counts before each fixed deadline, and joint correctness/time wins and losses. Use percentile 95% bootstrap intervals on paired differences of request means within short families by sampling whole topics, issuers or database schemas with replacement, retaining original multiplicities. Two locale clusters do not support credible sampling intervals. All paired requests and observed failures remain in each denominator. Native costs include registered retry charges. This analysis cannot change the registered decision, resolve hardware run-order confounding or establish reliability on a new task family.",
    }
    if freeze:
        committed(REG)
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    committed(DOC / "staged-query-reading.json")
    root = Path("research/results") / STAGE
    control = unpack(root / "control.json.gz")
    loop = unpack(root / "candidate.json.gz")
    data = []
    for w, a, b in zip(panel, loop, control, strict=True):
        assert w["id"] == a["id"] == b["id"]
        row = []
        for r in (a, b):
            q = r["requests"][0]
            row.append(
                (
                    int(q["score"]["success"]),
                    int(q["score"]["completed"]),
                    q["wall_s"],
                    q["first_s"],
                )
            )
        data.append(row)
    arr = np.asarray(data)
    output = {}
    deadlines = []
    intervals = []
    rng = np.random.default_rng(plan["seed"])
    for f, ix in {**families, "all": list(range(len(panel)))}.items():
        x = arr[ix]
        scores = x[:, :, 0]
        times = x[:, :, 2]
        summary = {"n": len(ix), "latency": {}}
        for a, arm in enumerate(("loop", "base")):
            summary["latency"][arm] = {
                m: {
                    "p50": float(np.median(x[:, a, j])),
                    "p95": float(np.quantile(x[:, a, j], 0.95)),
                }
                for j, m in enumerate(METRICS)
                if j >= 2
            }
            for t in DEADLINES:
                on_time = times[:, a] <= t
                deadlines.append(
                    {
                        "family": f,
                        "arm": arm,
                        "seconds": t,
                        "n": len(ix),
                        "correct": int((scores[:, a] * on_time).sum()),
                        "completed": int((x[:, a, 1] * on_time).sum()),
                    }
                )
        summary["loop_gained_and_faster"] = int(
            ((scores[:, 0] > scores[:, 1]) & (times[:, 0] < times[:, 1])).sum()
        )
        summary["loop_lost_and_slower"] = int(
            ((scores[:, 0] < scores[:, 1]) & (times[:, 0] > times[:, 1])).sum()
        )
        summary["both_correct"] = int(((scores[:, 0] == 1) & (scores[:, 1] == 1)).sum())
        summary["faster_on_both_correct"] = int(
            (
                (scores[:, 0] == 1) & (scores[:, 1] == 1) & (times[:, 0] < times[:, 1])
            ).sum()
        )
        output[f] = summary
        if f in ("all", "locale"):
            continue
        groups = defaultdict(list)
        for j, k in enumerate(ix):
            groups[panel[k]["cluster"]].append(j)
        d = x[:, 0] - x[:, 1]
        group_sums = np.array([d[js].sum(0) for js in groups.values()])
        group_n = np.array([len(js) for js in groups.values()])
        samples = rng.integers(len(groups), size=(plan["draws"], len(groups)))
        reps = group_sums[samples].sum(1) / group_n[samples].sum(1)[:, None]
        for j, m in enumerate(METRICS):
            lo, hi = np.quantile(reps[:, j], [0.025, 0.975])
            intervals.append(
                {
                    "family": f,
                    "metric": m,
                    "source_clusters": len(groups),
                    "n": len(ix),
                    "loop_minus_base": float(d[:, j].mean()),
                    "lower": float(lo),
                    "upper": float(hi),
                }
            )
    write_csv(Path("paper/staged-query-deadlines.csv"), deadlines)
    write_csv(Path("paper/staged-query-source-intervals.csv"), intervals)
    persist(
        DOC / "staged-query-distribution.json",
        {
            "plan": digest(PLAN),
            "archive": digest(root / "checksums.json"),
            "families": output,
            "intervals": intervals,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "report"))
    run(p.parse_args().mode == "freeze")
