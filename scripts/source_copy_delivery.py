"""Analyze paired source clusters, complete deliveries and timing without retraining or tuning."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_primary import ARMS, costs
from scripts.assistance_deliberation import write_csv
from scripts.register_source_copy import RESULT, PAIR, PROTOCOL
from scripts.source_copy_material import PANEL
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/source-copy-delivery.json")
OUT = Path("docs/measurements/source-copy-delivery.json")
P = Path("paper")


def run(freeze):
    spec = dict(
        protocol=digest(PROTOCOL),
        panel=digest(PANEL),
        script=digest(Path(__file__)),
        seed=362713,
        comparisons="After all blinded ratings are sealed, report per-request elapsed wall time, first wait, peak native allocation, opening throughput, post-first output rate, emitted length, strict success and terminal status. For each of reading and financial dialogues, bootstrap 10000 complete source clusters with replacement, keeping first/live/matched presentations and both arms together. Report percentile intervals on live pair completion, live pair correctness, common-history follow-up correctness, all-three-presentation correctness and mean source wall time. The preregistered exact paired tests remain primary. Report conditional follow-up success when the previous free reply succeeded/failed, with denominators, and the frequency of recovered or newly lost answers under own vs fixed history. Count requests both strictly correct and better on all five costs, alongside each individual regression.",
    )
    if freeze:
        persist(PLAN, spec)
        return
    committed(PLAN)
    assert read(PLAN) == spec
    root = RESULT / PAIR
    committed(root / "checksums.json")
    if not (root / "control.json.gz").is_file():
        return
    from scripts.source_copy_reading import GRADES

    committed(GRADES)
    panel = unpack(PANEL)
    runs = {arm: unpack(root / (arm + ".json.gz")) for arm in ARMS}
    requests = []
    sources = []
    cond = Counter()
    for i, w in enumerate(panel):
        source = dict(id=w["id"], family=w["family"], cluster=w["cluster"])
        for j, label in enumerate(("first", "live", "matched")):
            row = dict(
                id=w["id"], family=w["family"], cluster=w["cluster"], presentation=label
            )
            qa = []
            for arm in ARMS:
                r = runs[arm][i]
                assert w["id"] == r["id"]
                q = r["requests"][j]
                assert label == q["history"]
                c = costs([q])
                tr = q["trajectory"]
                row.update({arm + "_" + k: v for k, v in c.items()})
                row.update(
                    {
                        arm + "_correct": q["score"]["success"],
                        arm + "_status": tr["status"],
                        arm + "_opening": q["opening"],
                    }
                )
                source[arm + "_" + label + "_correct"] = q["score"]["success"]
                source[arm + "_" + label + "_done"] = tr["status"] == "answered"
                source[arm + "_wall"] = source.get(arm + "_wall", 0) + q["wall_s"]
                qa.append(q)
            better = all(
                row["candidate_" + k] < row["control_" + k]
                for k in ("wall_s", "first_s", "peak")
            ) and all(
                row["candidate_" + k] > row["control_" + k]
                for k in ("opening_tps", "decode_tps")
            )
            row["better_all_costs"] = better
            row["better_all_costs_and_correct"] = better and row["candidate_correct"]
            requests.append(row)
        for arm in ARMS:
            first = source[arm + "_first_correct"]
            live = source[arm + "_live_correct"]
            matched = source[arm + "_matched_correct"]
            cond[(w["family"], arm, first, live, matched)] += 1
            source[arm + "_live_pair"] = first and live
            source[arm + "_all_three"] = first and live and matched
            source[arm + "_live_completed"] = (
                source[arm + "_first_done"] and source[arm + "_live_done"]
            )
        sources.append(source)
    rng = np.random.default_rng(spec["seed"])
    intervals = []
    for group in ("reading", "report"):
        cohort = [w for w in sources if w["family"] == group]
        n = len(cohort)
        ix = rng.integers(n, size=(10000, n))
        for metric in (
            "live_pair",
            "live_completed",
            "matched_correct",
            "all_three",
            "wall",
        ):
            d = np.array(
                [
                    float(w["candidate_" + metric]) - float(w["control_" + metric])
                    for w in cohort
                ]
            )
            lo, hi = np.quantile(d[ix].mean(axis=1), [0.025, 0.975])
            intervals.append(
                dict(
                    family=group,
                    metric=metric,
                    n=n,
                    difference=d.mean(),
                    lower=lo,
                    upper=hi,
                )
            )
    write_csv(P / "source-copy-requests.csv", requests)
    write_csv(P / "source-copy-conversations.csv", sources)
    write_csv(P / "source-copy-intervals.csv", intervals)
    conditions = [
        dict(
            family=k[0],
            arm=k[1],
            first_correct=k[2],
            live_correct=k[3],
            matched_correct=k[4],
            sources=n,
        )
        for k, n in sorted(cond.items())
    ]
    write_csv(P / "source-copy-history-transitions.csv", conditions)
    persist(
        OUT,
        dict(
            plan=digest(PLAN),
            archive=digest(root / "checksums.json"),
            requests=len(requests),
            source_clusters=len(sources),
            both_faster_and_correct=sum(
                r["better_all_costs_and_correct"] for r in requests
            ),
            all_costs=sum(r["better_all_costs"] for r in requests),
            differences=intervals,
            conditional=conditions,
            files={
                k: digest(P / ("source-copy-" + k + ".csv"))
                for k in (
                    "requests",
                    "conversations",
                    "intervals",
                    "history-transitions",
                )
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "measure"))
    run(parser.parse_args().action == "freeze")
