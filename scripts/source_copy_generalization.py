"""Retrospective equal-domain delivery accuracy from the committed anonymous assessments."""

from pathlib import Path

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.research.conditional_primary import ARMS
from scripts.assistance_deliberation import write_csv
from scripts.source_copy_material import PANEL
from scripts.source_copy_reading import PLAN, GRADES, OUT as REVIEW
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed


def run():
    for p in (PLAN, GRADES, REVIEW):
        committed(p)
    judged = read(GRADES)["cases"]
    rows = []
    for w in unpack(PANEL):
        key = fingerprint([digest(PLAN), w["id"], "source"])[:12]
        first = ARMS[int(fingerprint([w["id"], digest(PLAN), "side"])[-4:], 16) % 2]
        order = (first, next(a for a in ARMS if a != first))
        for arm, grades in zip(order, judged[key], strict=True):
            assert [g["history"] for g in grades] == ["first", "live", "matched"]
            rows.append(
                dict(
                    source=w["id"],
                    family=w["family"],
                    arm=arm,
                    presentations=len(grades),
                    correct=sum(g["correct"] for g in grades),
                    supported=sum(
                        g["correct"] and g["supported"] and not g["unsupported"]
                        for g in grades
                    ),
                    usable=sum(g["usable"] for g in grades),
                )
            )
    # Recover the existing signed source-level judgments without regrading an answer.
    review = read(REVIEW)
    for family in ("reading", "report"):
        for arm in ARMS:
            group = [r for r in rows if (r["family"], r["arm"]) == (family, arm)]
            assert len(group) == 24
            assert sum(r["correct"] for r in group) == sum(
                review["positions"][f"{family}/{h}/{arm}"]["correct"]
                for h in ("first", "live", "matched")
            )
            assert (
                sum(r["usable"] == 3 for r in group)
                == review["families"][family]["usable"][arm]
            )
    write_csv(Path("paper/source-copy-supported-delivery.csv"), rows)
    rng = np.random.default_rng(713941)
    domains = {}
    draws = []
    for family in ("reading", "report"):
        by_arm = [
            {r["source"]: r for r in rows if r["arm"] == arm and r["family"] == family}
            for arm in ARMS
        ]
        assert set(by_arm[0]) == set(by_arm[1])
        pairs = np.array(
            [
                [arm[k]["supported"] / arm[k]["presentations"] for arm in by_arm]
                for k in sorted(by_arm[0])
            ]
        )
        diffs = pairs[:, 0] - pairs[:, 1]
        draws.append(
            diffs[rng.integers(0, len(diffs), (20000, len(diffs)))].mean(axis=1)
        )
        domains[family] = dict(
            sources=len(diffs),
            scores=dict(zip(ARMS, pairs.mean(axis=0).tolist())),
            gains=int((diffs > 0).sum()),
            losses=int((diffs < 0).sum()),
            interval=np.quantile(draws[-1], [0.025, 0.975]).tolist(),
        )
    results = {
        arm: float(np.mean([d["scores"][arm] for d in domains.values()]))
        for arm in ARMS
    }
    out = dict(
        assessments=digest(GRADES),
        domains=domains,
        scores=results,
        difference=results["candidate"] - results["control"],
        interval=np.quantile(np.mean(draws, axis=0), [0.025, 0.975]).tolist(),
        method="Retrospective two-domain analysis. For each presentation require a correct delivered conclusion with adequate source support and no unsupported additions. Explicit tool execution is reported separately. Average the three presentations within each original source, then sources within each domain, then weight the two observed domains equally. Resample complete paired sources within each domain 20000 times. Measures this fixed cohort and one reviewer's judgments. Do not substitute scores from another checkpoint or impute unobserved capabilities.",
    )
    persist(Path("docs/measurements/source-copy-supported-delivery.json"), out)
    print(out)


if __name__ == "__main__":
    run()
