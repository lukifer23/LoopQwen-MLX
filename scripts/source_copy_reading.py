"""Publish blind source packets and score judgments committed before the identity is revealed."""

import argparse
from collections import defaultdict
from pathlib import Path
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_primary import ARMS
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.native_function_study import executions
from scripts.register_source_copy import PAIR, PROTOCOL, RESULT
from scripts.source_copy_material import PANEL
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

DIR = Path("docs/measurements/source-copy-reading")
MASK = DIR / "anonymous.json"
GRADES = DIR / "assessments.json"
OUT = DIR / "report.json"
PLAN = Path("research/diagnostics/source-copy-source-review.json")


def plan():
    return dict(
        comparison=digest(PROTOCOL),
        panel=digest(PANEL),
        implementation=digest(Path(__file__)),
        rubric="Each anonymous response is scored on actual delivered content. Read the full original story/report, the two user questions, available alternate human references and every real callback for both systems. For each of the two first answers, live follow-ups and fixed-history follow-ups per source, record correct delivered conclusion, adequate source support, unsupported material, and fully usable assessment. Reading can express the reference in different words. Financial delivery must give the requested direction, magnitude and correct source units. Full financial usability additionally requires an appropriate executed calculation with a correct returned amount. A missing or unfinished visible answer is unusable. Score explanations if offered. A direct answer alone is acceptable when it fully answers the question. Archive a source-specific reason for every rating. Score every case before exposing model identity or aggregate results. Source-level success requires all three presentations fully usable. Preserve all individual losses and source-level exact paired uncertainty.",
    )


def run(mode):
    if mode == "freeze":
        persist(PLAN, plan())
        return
    committed(PLAN)
    assert read(PLAN) == plan()
    root = RESULT / PAIR
    committed(root / "checksums.json")
    data = {arm: unpack(root / (arm + ".json.gz")) for arm in ARMS}
    rows = unpack(PANEL)
    cases = []
    mapping = {}
    for i, w in enumerate(rows):
        key = fingerprint([digest(PLAN), w["id"], "source"])[:12]
        left = ARMS[int(fingerprint([w["id"], digest(PLAN), "side"])[-4:], 16) % 2]
        order = [left, next(a for a in ARMS if a != left)]
        mapping[key] = order
        replies = []
        for arm in order:
            item = data[arm][i]
            assert item["id"] == w["id"]
            rr = []
            for req in item["requests"]:
                tr = req["trajectory"]
                rr.append(
                    dict(
                        history=req["history"],
                        delivered=tr["answer"],
                        status=tr["status"],
                        executions=executions(tr),
                    )
                )
            replies.append(rr)
        cases.append(
            dict(
                key=key,
                family=w["family"],
                source=w["body"],
                turns=w["turns"],
                responses=replies,
            )
        )
    cases.sort(key=lambda w: w["key"])
    if mode == "blind":
        persist(
            MASK,
            dict(
                plan=digest(PLAN), archive=digest(root / "checksums.json"), cases=cases
            ),
        )
        return
    committed(MASK)
    committed(GRADES)
    ratings = read(GRADES)
    assert read(MASK)["cases"] == cases and ratings["mask"] == digest(MASK)
    lookup = ratings["cases"]
    assert set(lookup) == set(mapping)
    groups = defaultdict(list)
    positions = defaultdict(list)
    for case in cases:
        key = case["key"]
        grades = lookup[key]
        order = mapping[key]
        assert len(grades) == 2
        pair = {}
        for j, (arm, rr, gs) in enumerate(
            zip(order, case["responses"], grades, strict=True)
        ):
            assert len(gs) == len(rr) == 3
            accepted = []
            for q, g in zip(rr, gs, strict=True):
                assert g["history"] == q["history"] and g["reason"].strip()
                assert all(
                    type(g[k]) is bool
                    for k in ("correct", "supported", "unsupported", "usable")
                )
                if not q["delivered"]:
                    assert not g["correct"] and not g["usable"]
                assert not g["usable"] or (
                    g["correct"]
                    and g["supported"]
                    and not g["unsupported"]
                    and (case["family"] != "report" or q["executions"])
                )
                accepted.append(g["usable"])
                positions[case["family"] + "/" + q["history"] + "/" + arm].append(g)
            pair[arm] = int(all(accepted))
        groups[case["family"]].append(pair)
    results = {}
    for family, pairs in groups.items():
        d = [p["candidate"] - p["control"] for p in pairs]
        results[family] = dict(
            n=len(d),
            usable={a: sum(p[a] for p in pairs) for a in ARMS},
            gains=d.count(1),
            losses=d.count(-1),
            p=exact_sign_flip(d),
        )
    gates = {
        f: r["gains"] - r["losses"] >= 5 and r["losses"] == 0 and r["p"] <= 0.05
        for f, r in results.items()
    }
    persist(
        OUT,
        dict(
            plan=digest(PLAN),
            mask=digest(MASK),
            grades=digest(GRADES),
            families=results,
            positions={
                k: {
                    col: sum(g[col] for g in vals)
                    for col in ("correct", "supported", "unsupported", "usable")
                }
                for k, vals in positions.items()
            },
            gates=gates,
            passed=all(gates.values()),
            joint=all(gates.values()) and read(root / "decision.json")["passed"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "blind", "score"))
    run(p.parse_args().action)
