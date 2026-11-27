"""Check whether free-form scientific reasoning survives a source reading of every authored pair."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.contrastive_transfer import ARMS, paired
from scripts.contrastive_transfer_panel import PANEL
from scripts.contrastive_transfer_reading import DOC, ROOT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/contrastive-qualitative-reading.json")
MASK = DOC / "contrastive-qualitative-blinded.json"
RATINGS = DOC / "contrastive-qualitative-blinded-ratings.json"
REPORT = DOC / "contrastive-qualitative-reading.json"


def run(mode):
    items = [w for w in unpack(PANEL)["rows"] if w["family"] == "qualitative"]
    assert len(items) == 48
    plan = {
        "panel": digest(PANEL),
        "ordered": [w["id"] for w in items],
        "selection": "All 48 authored questions, both members of each of 24 held qualitative backgrounds. Choose no answers based on outcome.",
        "scope": "Descriptive robustness to prose and negation alongside the frozen direct-choice scorer. A project reviewer sees the complete question, physical relation, original human answer, and hash-shuffled distinct visible replies. Grade correct, partial, wrong or absent, with a specific reason. Accept a correct explanation or a short correct answer regardless of where the choice occurs. A relevant contradiction prevents full credit. An answer that merely mentions the correct choice receives no credit. Seal all grades before revealing either arm or the primary aggregate. Recompute both individual changes and whole-source sign flips. This reading changes no original promotion criterion.",
    }
    if mode == "plan":
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    data = [
        {r["id"]: r["requests"][0] for r in unpack(ROOT / (a + ".json.gz"))}
        for a in ARMS
    ]
    mask = {}
    mapping = {}
    for w in items:
        k = w["id"]
        answers = [d[k]["trajectory"]["answer"] or "" for d in data]
        options = sorted(set(answers), key=lambda t: fingerprint([598237, k, t]))
        mask[k] = {
            "request": w["questions"][0],
            "reference": w["gold"],
            "choices": w["choices"],
            "replies": options,
        }
        mapping[k] = [options.index(x) for x in answers]
    if mode == "blind":
        persist(MASK, mask)
        return
    committed(MASK)
    committed(RATINGS)
    assert read(MASK) == mask
    ratings = read(RATINGS)
    assert ratings.keys() == mask.keys()
    counts = {a: Counter() for a in ARMS}
    groups = defaultdict(list)
    for w in items:
        k = w["id"]
        r = ratings[k]
        assert r["reason"].strip() and len(r["replies"]) == len(mask[k]["replies"])
        success = []
        for a, d, i in zip(ARMS, data, mapping[k], strict=True):
            g = r["replies"][i]
            rating = g["rating"]
            assert rating in ("fully usable", "partial", "wrong", "absent")
            assert (rating == "absent") == (not mask[k]["replies"][i].strip()) and type(
                g["unsupported"]
            ) is bool
            auto = d[k]["score"]["success"]
            correct = rating == "fully usable"
            success.append(correct)
            counts[a][rating] += 1
            counts[a]["unsupported"] += g["unsupported"]
            counts[a]["accepted_by_reading_only"] += correct and not auto
            counts[a]["accepted_by_automatic_only"] += auto and not correct
        groups[w["cluster"]].append(success)
    assert len(groups) == 24
    save(
        REPORT,
        {
            "plan": digest(PLAN),
            "ratings": digest(RATINGS),
            "source_pairs": paired(groups),
            "arms": counts,
        },
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("mode", choices=("plan", "blind", "report"))
    run(args.parse_args().mode)
