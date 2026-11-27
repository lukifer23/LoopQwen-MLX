"""Seal source-visible ratings for every changed table, database and open reply."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE
from scripts.integrated_assistance_reading import PLAN as EARLIER_PLAN

NAME = "table-policy-gradient-01"
ROOT = Path("research/results") / NAME
PLAN = Path("research/diagnostics/table-policy-reading.json")
DOC = Path("docs/measurements")
MASK = DOC / "table-policy-blinded.json"
RATINGS = DOC / "table-policy-blinded-ratings.json"
SCORES = DOC / "table-policy-reading.json"


def rows_and_plan():
    panel = read("research/data/table-policy-panel.json")
    criteria = read(EARLIER_PLAN)["criteria"]
    selected = [
        w for w in panel["rows"] if w["family"] in ("open", "database", "changed_table")
    ]
    criteria["changed_table"] = (
        "Read the question and the complete supplied table. Compare the full visible reply "
        "with the answer independently chosen by at least three source annotators and the "
        "actual cell evidence. Accept equivalent names, units and accurate natural sentences. "
        "If the source gold is ambiguous, grade the actual requested fact, give the row or "
        "operation supporting the judgment and explain any disagreement. Fully usable means "
        "the requested answer is clear and accurate without an unsupported material claim. "
        "An answer that includes incompatible alternatives is not fully usable. "
        "Use partial for genuinely relevant incomplete help, wrong for a material error, "
        "absent for an empty reply. Judge each version independently."
    )
    plan = {
        "ordered_requests": [w["id"] for w in selected],
        "criteria": criteria,
        "release": "Retain all registered integrated open/database thresholds. For changed tables "
        "require at least six whole pairs semantically correct, a net five pairs and exact source-paired "
        "two-sided p <= .05, positive net in both versions, zero individual fully usable Base replies "
        "lost and no additional materially unsupported changed-table replies. Report the exact "
        "grader separately from this source-visible reading. Either gate can stop promotion.",
        "format": "Only the visible replies, request, reference and source are shown, in per-request "
        "hash order. Identical replies collapse to one. Commit one grade and unsupported flag for "
        "each distinct reply, a tie or preferred offset and a request-specific reason before mapping arms.",
        "source": digest(SOURCE),
    }
    assert Counter(w["family"] for w in selected) == {
        "changed_table": 48,
        "database": 24,
        "open": 12,
    }
    return selected, plan


def mapping(selected):
    raw = {
        arm: {
            r["id"]: r["requests"][0]["trajectory"]["answer"] or ""
            for r in unpack(ROOT / (arm + ".json.gz"))
        }
        for arm in ("candidate", "control")
    }
    tables = {w["id"]: w for w in read(SOURCE)["cohorts"]["dev"]}
    mask, order = {}, {}
    for w in selected:
        key = w["id"]
        replies = sorted(
            {r[key] for r in raw.values()}, key=lambda t: fingerprint([842115, key, t])
        )
        rec = {"request": w["questions"][0], "replies": replies, "family": w["family"]}
        if w["family"] == "changed_table":
            (v,) = [
                v for v in tables[w["row_id"]]["versions"] if v["world"] == w["world"]
            ]
            assert w["gold"] == v["answer"]
            rec.update(rows=v["rows"], reference=v["answer"])
        elif w["family"] == "database":
            rec.update(reference=w["human_answer"], expected_value=w["expected_value"])
        else:
            rec["reference"] = w["human_reply"]
        mask[key] = rec
        order[key] = {a: replies.index(r[key]) for a, r in raw.items()}
    return mask, order


def committed(path):
    content = subprocess.check_output(["git", "show", "HEAD:" + str(path)])
    assert path.read_bytes() == content, f"Seal {path} in git first"


def run(mode):
    selected, plan = rows_and_plan()
    if mode == "plan":
        save(PLAN, plan)
        return
    assert plan == read(PLAN)
    mask, order = mapping(selected)
    if mode == "blind":
        if MASK.exists():
            assert mask == read(MASK)
        else:
            save(MASK, mask)
        return
    from scripts.assistance_uncertainty import exact_sign_flip

    committed(RATINGS)
    committed(MASK)
    assert mask == read(MASK)
    ratings = read(RATINGS)
    assert mask.keys() == ratings.keys()
    counts = {a: defaultdict(Counter) for a in ("candidate", "control")}
    grouped = defaultdict(list)
    variants = defaultdict(list)
    lost = 0
    for w in selected:
        k, f = w["id"], w["family"]
        j = ratings[k]
        replies = mask[k]["replies"]
        assert len(j["replies"]) == len(replies) and j["reason"].strip()
        assert j["preferred"] in ["tie", *range(len(replies))]
        if len(replies) == 1:
            assert j["preferred"] == "tie"
        success = []
        for arm, i in order[k].items():
            g = j["replies"][i]
            assert type(g["unsupported"]) is bool and g["rating"] in (
                "fully usable",
                "partial",
                "wrong",
                "absent",
            )
            assert (g["rating"] == "absent") == (not replies[i].strip())
            rec = counts[arm][f]
            rec[g["rating"]] += 1
            rec["unsupported"] += g["unsupported"]
            rec["win"] += j["preferred"] == i
            rec["tie"] += j["preferred"] == "tie"
            success.append(g["rating"] == "fully usable")
        if f == "changed_table":
            grouped[w["cluster"]].append(success)
            variants["original" if w["world"] == 0 else "altered"].append(
                int(success[0]) - int(success[1])
            )
            lost += success == [False, True]
    whole = [
        [all(pair[arm] for pair in versions) for arm in (0, 1)]
        for versions in grouped.values()
    ]
    assert len(whole) == 24 and all(len(v) == 2 for v in grouped.values())
    p = exact_sign_flip([int(a) - int(b) for a, b in whole])
    ca, co = (counts[a] for a in ("candidate", "control"))
    whole_counts = list(map(sum, zip(*whole, strict=True)))
    gates = {
        "open": ca["open"]["win"] >= max(6, co["open"]["win"] + 1)
        and ca["open"]["fully usable"] >= co["open"]["fully usable"]
        and ca["open"]["unsupported"] <= co["open"]["unsupported"],
        "database": ca["database"]["fully usable"] >= co["database"]["fully usable"],
        "changed_table": whole_counts[0] >= max(6, whole_counts[1] + 5)
        and p <= 0.05
        and all(sum(v) > 0 for v in variants.values())
        and not lost
        and ca["changed_table"]["unsupported"] <= co["changed_table"]["unsupported"],
    }
    save(
        SCORES,
        {
            "arms": counts,
            "whole_tables": whole_counts,
            "p": p,
            "lost_individual": lost,
            "by_version_net": {k: sum(v) for k, v in variants.items()},
            "gates": gates,
            "reading_gate": all(gates.values()),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "blind", "report"))
    run(parser.parse_args().mode)
