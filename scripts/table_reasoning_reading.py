"""Post-comparison masked source review of the apparent gain in human reasoning F1."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.linked_rollin import exact_sign_flip
from scripts.table_policy_reading import ROOT, committed

PLAN = Path("research/diagnostics/table-reasoning-reading.json")
DOC = Path("docs/measurements")
MASK = DOC / "table-reasoning-blinded.json"
RATINGS = DOC / "table-reasoning-blinded-ratings.json"
RESULT = DOC / "table-reasoning-source-reading.json"
ARMS = ("candidate", "control")


def selected():
    panel = read("research/data/table-policy-panel.json")["rows"]
    rows = []
    for category in ("conversation", "causal"):
        pool = [
            r
            for r in panel
            if r["family"]
            in (
                ("human_first", "human_followup")
                if category == "conversation"
                else ("effects",)
            )
        ]
        groups = sorted(
            {w["cluster"] for w in pool},
            key=lambda k: fingerprint([804932, category, k]),
        )
        assert len(groups) == 24
        chosen = [r for r in pool if r["cluster"] in set(groups[:12])]
        assert len(chosen) == 24 and set(
            Counter(r["cluster"] for r in chosen).values()
        ) == {2}
        rows.extend(chosen)
    return sorted(rows, key=lambda r: fingerprint([804932, "presentation", r["id"]]))


def run(mode):
    rows = selected()
    plan = {
        "question": "Do higher automatic causal and first-question token-overlap scores reflect correct human answers when the full source, question and previous turn are examined?",
        "scope": "Post hoc after inspecting aggregate F1 and automatic correctness. No additional model generation. Choose 12 of the 24 distinct backgrounds by hash in each of QuAC and ROPES. Include both previously generated requests per background. Do not choose using any request-level win, loss, completion, length or model reply.",
        "review": "Show original prompt, source annotations and hash-shuffled distinct visible answers. Check the comparison or requested fact from the actual passage and situation. Respect first-person references in dialogues. Credit equivalent wording and honest unanswerability when the source does not support the requested claim. Grade correct, partial, wrong or absent with one request-specific rationale identifying the deciding evidence. A correct short answer need not include hidden reasoning. A material extra factual error prevents full credit. Flag unsupported additions. The same project reviewer knows the earlier aggregate result. This is a descriptive sensitivity study and does not change any release gate.",
        "selection_seed": 804932,
        "ordered": [r["id"] for r in rows],
        "primary": digest(ROOT / "checksums.json"),
    }
    if mode == "plan":
        if PLAN.exists():
            assert read(PLAN) == plan
        else:
            save(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    data = {
        a: {r["id"]: r["requests"][0] for r in unpack(ROOT / (a + ".json.gz"))}
        for a in ARMS
    }
    mask = {}
    mapping = {}
    for r in rows:
        k = r["id"]
        answers = [data[a][k]["trajectory"]["answer"] or "" for a in ARMS]
        options = sorted(set(answers), key=lambda t: fingerprint([185083, k, t]))
        mask[k] = {
            "family": r["family"],
            "request": r["questions"][0],
            "reference": r["annotations"] if "annotations" in r else r["answers"],
            "replies": options,
        }
        mapping[k] = [options.index(x) for x in answers]
    if mode == "blind":
        if MASK.exists():
            assert read(MASK) == mask
        else:
            save(MASK, mask)
        return
    committed(RATINGS)
    committed(MASK)
    assert mask == read(MASK)
    grades = read(RATINGS)
    assert grades.keys() == mask.keys()
    counts = {a: defaultdict(Counter) for a in ARMS}
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = r["id"]
        f = r["family"]
        rec = grades[k]
        assert rec["reason"].strip() and len(rec["replies"]) == len(mask[k]["replies"])
        both = []
        for a, i in zip(ARMS, mapping[k], strict=True):
            g = rec["replies"][i]
            rating = g["rating"]
            text = mask[k]["replies"][i]
            assert (
                (rating == "absent") == (not text.strip())
                and rating in ("fully usable", "partial", "wrong", "absent")
                and type(g["unsupported"]) is bool
            )
            stat = counts[a][f]
            stat[rating] += 1
            stat["unsupported"] += g["unsupported"]
            automatic = data[a][k]["score"]
            stat["automatic_exact"] += automatic["success"]
            stat["f1_sum"] += automatic["f1"]
            both.append(rating == "fully usable")
        groups["causal" if f == "effects" else "conversation"][r["cluster"]].append(
            both
        )
    result = {}
    for name, sets in groups.items():
        assert len(sets) == 12 and all(len(v) == 2 for v in sets.values())
        choices = [[all(p[i] for p in v) for i in range(2)] for v in sets.values()]
        deltas = [int(a) - int(b) for a, b in choices]
        result[name] = {
            "whole": [sum(pair[i] for pair in choices) for i in (0, 1)],
            "whole_exact_p": exact_sign_flip(deltas),
            "individual_gains": sum(a and not b for v in sets.values() for a, b in v),
            "individual_losses": sum(b and not a for v in sets.values() for a, b in v),
        }
    save(
        RESULT,
        {
            "counts": counts,
            "sources": result,
            "ratings": digest(RATINGS),
            "plan": digest(PLAN),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "blind", "report"])
    run(parser.parse_args().mode)
