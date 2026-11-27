"""Mask every short source response and require committed judgments before unmasking."""

import argparse
from collections import Counter, defaultdict
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


def run(mode, *, repair=False, staged=False, learner=False, content=False, bounded=False):
    panel, plan, stage = PANEL, PLAN, PRIMARY
    mask_path, rating_path, report_path = MASK, RATINGS, REPORT
    if learner:
        from scripts.learner_prefix_panel import PANEL as panel, PLAN as plan
        from scripts.register_learner_prefix import PRIMARY as stage

        mask_path = DOC / "learner-prefix-blinded.json"
        rating_path = DOC / "learner-prefix-blinded-ratings.json"
        report_path = DOC / "learner-prefix-reading.json"
    if content:
        from scripts.register_content_read import PANEL as panel, PLAN as plan, PRIMARY as stage

        mask_path = DOC / "content-read-blinded.json"
        rating_path = DOC / "content-read-blinded-ratings.json"
        report_path = DOC / "content-read-reading.json"
    if bounded:
        from scripts.register_bounded_read import PANEL as panel, PLAN as plan, PRIMARY as stage

        mask_path = DOC / "bounded-read-blinded.json"
        rating_path = DOC / "bounded-read-blinded-ratings.json"
        report_path = DOC / "bounded-read-reading.json"
    if repair or staged:
        from scripts.register_repair_query import (
            PANEL as panel,
        )
        from scripts.register_repair_query import (
            PLAN as plan,
        )
        from scripts.register_repair_query import (
            PRIMARY as stage,
        )

        prefix = "staged-query" if staged else "repair-query"
        if staged:
            stage = "staged-query-transfer-01"
        mask_path = DOC / (prefix + "-blinded.json")
        rating_path = DOC / (prefix + "-blinded-ratings.json")
        report_path = DOC / (prefix + "-reading.json")
    assert read(plan)["material"] == digest(panel)
    rows = [w for w in unpack(panel)["rows"] if w["family"] != "locale"]
    data = {
        a: {
            r["id"]: r["requests"][0]
            for r in unpack(RESULTS / stage / (a + ".json.gz"))
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
        if w["family"] == "field":
            mask[k].update(
                condition=w["condition"],
                history=w["messages"],
                interpretation=w["interpretation"],
                expert_evidence=w["expert_result"],
                source_split=w["source_split"],
            )
    if mode == "blind":
        persist(mask_path, mask)
        return
    committed(mask_path)
    committed(rating_path)
    grades = read(rating_path)
    assert read(mask_path) == mask and grades.keys() == mask.keys()
    out = {}
    for fam in dict.fromkeys(w["family"] for w in rows):
        stats = {a: Counter() for a in ARMS}
        pairs = []
        schema = defaultdict(list)
        conditions = defaultdict(list)
        wording = defaultdict(list)
        splits = defaultdict(list)
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
                if fam == "field":
                    if w["interpretation"]["tag"] == "no_specific_ambiguity_flagged":
                        usable = usable and score["grounded"]
                    elif w["condition"] != "valid_current":
                        usable = usable and score["last_successful_is_fresh"]
                values.append(int(usable))
                c = stats[arm]
                c[g["rating"]] += 1
                c["grounded_usable"] += usable
                c["unsupported"] += g["unsupported"]
                c["preferred"] += r["preferred"] == i
                c["tie"] += r["preferred"] == "tie"
            pairs.append(values)
            if fam == "field":
                delta = values[0] - values[1]
                schema[w["source"]].append(delta)
                conditions[w["condition"]].append(delta)
                wording[w["interpretation"]["tag"]].append(delta)
                splits[w["source_split"]].append(delta)
        diff = [a - b for a, b in pairs]
        p = exact_sign_flip([sum(v) for v in schema.values()] if schema else diff)
        ok = (
            all(sum(v) > 0 for v in conditions.values())
            and sum(diff) >= 6
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
        if schema:
            out[fam]["net_by_schema"] = {k: sum(v) for k, v in schema.items()}
            for key, group in (
                ("condition", conditions),
                ("wording", wording),
                ("split", splits),
            ):
                out[fam][key] = {
                    k: {"gains": v.count(1), "losses": v.count(-1), "n": len(v)}
                    for k, v in group.items()
                }
    save(
        report_path,
        {
            "passed": all(v["passed"] for v in out.values()),
            "families": out,
            "plan": digest(plan),
            "ratings": digest(rating_path),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("blind", "report"))
    cohort = p.add_mutually_exclusive_group()
    cohort.add_argument("--repair", action="store_true")
    cohort.add_argument("--staged", action="store_true")
    cohort.add_argument("--learner", action="store_true")
    cohort.add_argument("--content", action="store_true")
    cohort.add_argument("--bounded", action="store_true")
    a = p.parse_args()
    run(a.mode, repair=a.repair, staged=a.staged, learner=a.learner, content=a.content, bounded=a.bounded)
