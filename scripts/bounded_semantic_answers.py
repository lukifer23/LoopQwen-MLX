"""Assess answers to the original questions independently of the first-token and tool requirements."""

import argparse
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.research.native_function_study import executions
from scripts.register_bounded_read import PANEL, PRIMARY
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/bounded-read-semantic-answers.json")
DOC = Path("docs/measurements")


def run(freeze):
    committed(PANEL)
    spec = dict(
        panel=digest(PANEL),
        primary=PRIMARY,
        protocol="Before either arm's results or identities are inspected, read each distinct anonymous reply against the original question and source. Separately label its delivered conclusion as correct, incorrect or unclear. For reasoning accept an unambiguously expressed yes/no relation anywhere in the reply; the separate funded explanation rubric still requires the supporting reasoning. For finance require the correct sign, result and requested display scale/units, regardless of whether a real calculator was used. Contextually implicit units are acceptable if uniquely determined by the question and delivered numerical value. A reply with a wrong explicit unit is incorrect. Absent or internally unresolved replies are unclear. Unsupported extra claims remain separately recorded. Replay every arm's mapping only after the combined grades and the primary reading are committed. Publish counts alongside, leaving all registered source and product acceptance thresholds fixed.",
    )
    if freeze:
        persist(PLAN, spec)
        return
    committed(PLAN)
    assert read(PLAN) == spec
    ratings = DOC / "bounded-read-blinded-ratings.json"
    mask = DOC / "bounded-read-blinded.json"
    for p in (ratings, mask, DOC / "bounded-read-reading.json"):
        committed(p)
    rr = read(ratings)
    mm = read(mask)
    root = Path("research/results") / PRIMARY
    totals = {}
    for arm in ("candidate", "control"):
        c = {}
        for row in unpack(root / (arm + ".json.gz")):
            k = row["id"]
            q = row["requests"][0]
            t = q["trajectory"]
            opt = {"answer": t["answer"] or "", "calls": executions(t)}
            i = mm[k]["replies"].index(opt)
            label = rr[k]["replies"][i]["semantic"]
            assert label in ("correct", "incorrect", "unclear") and (
                opt["answer"].strip() or label == "unclear"
            )
            count = c.setdefault(row["family"], Counter())
            count[label] += 1
            count["strict_without_semantic"] += (
                q["score"]["success"] and label != "correct"
            )
            count["semantic_without_strict"] += (
                label == "correct" and not q["score"]["success"]
            )
        totals[arm] = c
    persist(
        DOC / "bounded-read-semantic-answers.json",
        dict(
            plan=digest(PLAN),
            ratings=digest(ratings),
            archive=digest(root / "checksums.json"),
            totals=totals,
        ),
    )
    print(totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "report"))
    run(p.parse_args().mode == "freeze")
