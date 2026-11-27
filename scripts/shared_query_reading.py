"""Mask all 80 complete source responses and require committed judgments before unmasking."""

import argparse
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_primary import ARMS
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.native_function_study import executions
from loopqwen.research.shared_query_features import PRIMARY, RESULTS
from scripts.register_shared_query import PANEL, PLAN
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

DOC = Path("docs/measurements")
MASK = DOC / "shared-query-blinded.json"
RATINGS = DOC / "shared-query-blinded-ratings.json"
REPORT = DOC / "shared-query-reading.json"


def run(mode):
    assert read(PLAN)["material"] == digest(PANEL)
    rows = [w for w in unpack(PANEL)["rows"] if w["family"] != "locale"]
    data = {
        a: {
            r["id"]: r["requests"][0]
            for r in unpack(RESULTS / PRIMARY / (a + ".json.gz"))
        }
        for a in ARMS
    }
    mask = {}
    mapping = {}
    for w in rows:
        k = w["id"]
        scores = [data[a][k] for a in ARMS]
        options = [
            {
                "answer": q["trajectory"]["answer"] or "",
                "calls": executions(q["trajectory"]),
            }
            for q in scores
        ]
        distinct = {fingerprint(z): z for z in options}
        unique = sorted(
            distinct.values(), key=lambda z: fingerprint(["source review", k, z])
        )
        mapping[k] = [unique.index(z) for z in options]
        mask[k] = {
            "family": w["family"],
            "source": w["cluster"],
            "request": w["questions"][0],
            "reference": w["reference"],
            "deciding_evidence": w.get("facts", w.get("evidence")),
            "reasoning": w.get("decomposition", w.get("program")),
            "replies": unique,
        }
    if mode == "blind":
        persist(MASK, mask)
        return
    committed(MASK)
    committed(RATINGS)
    grades = read(RATINGS)
    assert read(MASK) == mask and grades.keys() == mask.keys()
    out = {}
    for fam in ("strategy", "report"):
        stats = {a: Counter() for a in ARMS}
        pairs = []
        for w in rows:
            if w["family"] != fam:
                continue
            k = w["id"]
            r = grades[k]
            opt = mask[k]["replies"]
            assert len(r["replies"]) == len(opt) and r["reason"].strip()
            assert r["preferred"] == "tie" or (
                type(r["preferred"]) is int
                and 0 <= r["preferred"] < len(opt)
                and len(opt) > 1
            )
            values = []
            for arm, i in zip(ARMS, mapping[k], strict=True):
                g = r["replies"][i]
                score = data[arm][k]["score"]
                assert g["rating"] in ("fully usable", "partial", "wrong", "absent")
                assert (not opt[i]["answer"].strip()) == (g["rating"] == "absent")
                assert type(g["unsupported"]) is bool and not (
                    g["unsupported"] and g["rating"] == "fully usable"
                )
                usable = g["rating"] == "fully usable" and (
                    fam != "report" or score["arithmetic_path"]
                )
                values.append(int(usable))
                c = stats[arm]
                c[g["rating"]] += 1
                c["grounded_usable"] += usable
                c["unsupported"] += g["unsupported"]
                c["preferred"] += r["preferred"] == i
                c["tie"] += r["preferred"] == "tie"
            pairs.append(values)
        diff = [a - b for a, b in pairs]
        p = exact_sign_flip(diff)
        ok = (
            sum(diff) >= 6
            and -1 not in diff
            and p <= 0.05
            and stats["candidate"]["unsupported"] <= stats["control"]["unsupported"]
        )
        out[fam] = {
            "passed": ok,
            "gains": diff.count(1),
            "losses": diff.count(-1),
            "p": p,
            "arms": stats,
        }
    save(
        REPORT,
        {
            "passed": all(v["passed"] for v in out.values()),
            "families": out,
            "plan": digest(PLAN),
            "ratings": digest(RATINGS),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("blind", "report"))
    run(p.parse_args().mode)
