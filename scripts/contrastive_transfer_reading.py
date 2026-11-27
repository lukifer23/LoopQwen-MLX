"""Mask service conversations and verify the reviewed evidence against the real returned rows."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.contrastive_transfer import ARMS, PRIMARY, paired
from loopqwen.research.native_function_study import executions
from scripts.contrastive_transfer_panel import PANEL, PLAN
from scripts.program_language_material import norm
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ROOT = Path("research/results") / PRIMARY
DOC = Path("docs/measurements")
MASK = DOC / "contrastive-services-blinded.json"
RATINGS = DOC / "contrastive-services-blinded-ratings.json"
REPORT = DOC / "contrastive-services-reading.json"


def run(mode, *, streamed=False, variant=None):
    variant = variant or ("streamed" if streamed else "contrastive")
    if variant not in ("contrastive", "streamed", "elastic"):
        raise ValueError("Unknown registered comparison")
    root = ROOT.parent / (variant + "-world-transfer-01")
    prefix = variant + "-services"
    mask_file = DOC / (prefix + "-blinded.json")
    rating_file = DOC / (prefix + "-blinded-ratings.json")
    report_file = DOC / (prefix + "-reading.json")
    if mode == "report":
        for group in ("services", "qualitative"):
            prefix = variant + "-" + group
            committed(DOC / (prefix + "-blinded-ratings.json"))
    plan = read(PLAN)
    assert digest(PANEL) == plan["material"]
    panel = {r["id"]: r for r in unpack(PANEL)["rows"]}
    runs = {
        a: {r["id"]: r["requests"][0] for r in unpack(root / (a + ".json.gz"))}
        for a in ARMS
    }
    masked = {}
    mapping = {}
    for k in plan["masked_services"]:
        w = panel[k]
        answers = [runs[a][k]["trajectory"]["answer"] or "" for a in ARMS]
        options = sorted(set(answers), key=lambda t: fingerprint([692861, k, t]))
        masked[k] = {
            "request": w["questions"][0],
            "columns": w["rows"][0],
            "source_rows": w["rows"][1:],
            "original_reply": w["gold"],
            "replies": options,
        }
        mapping[k] = [options.index(t) for t in answers]
    assert len(masked) == 48
    if mode == "blind":
        persist(mask_file, masked)
        return
    committed(mask_file)
    committed(rating_file)
    assert read(mask_file) == masked
    grades = read(rating_file)
    assert grades.keys() == masked.keys()
    groups = defaultdict(list)
    counts = {a: Counter() for a in ARMS}
    grades_audit = []
    for k, r in grades.items():
        m = masked[k]
        assert r["reason"].strip() and len(r["replies"]) == len(m["replies"])
        grounded = []
        for arm, offset in zip(ARMS, mapping[k], strict=True):
            g = r["replies"][offset]
            grade = g["rating"]
            assert (
                grade in ("fully usable", "partial", "wrong", "absent")
                and (grade == "absent") == (not m["replies"][offset].strip())
                and type(g["unsupported"]) is bool
            )
            observed = [
                row
                for e in executions(runs[arm][k]["trajectory"])
                if e["call"]["name"] == "database_query"
                and e["observation"].get("referenced_tables") == ["records"]
                and not e["observation"].get("truncated", True)
                for row in e["observation"].get("rows", [])
            ]
            supported = False
            if grade == "fully usable":
                assert (
                    not g["unsupported"]
                    and type(g["row"]) is int
                    and 0 <= g["row"] < len(m["source_rows"])
                )
                fields = set(g["fields"])
                assert len(fields) >= 2 and fields <= set(m["columns"])
                source = m["source_rows"][g["row"]]
                cells = {
                    norm(str(source[m["columns"].index(f)]))
                    for f in fields
                    if source[m["columns"].index(f)] is not None
                }
                cells.difference_update({"", "true", "false", "yes", "no"})
                assert len(cells) >= 2
                supported = any(
                    len({norm(str(x)) for x in values if x is not None} & cells) >= 2
                    for values in observed
                )
            grounded.append(supported)
            counts[arm][grade] += 1
            counts[arm]["unsupported"] += g["unsupported"]
            counts[arm]["grounded"] += supported
            grades_audit.append(
                {
                    "id": k,
                    "arm": arm,
                    "rating": grade,
                    "grounded": supported,
                    "unsupported": g["unsupported"],
                    "row": g.get("row"),
                    "fields": g.get("fields", []),
                    "reason": r["reason"],
                }
            )
        groups[panel[k]["cluster"]].append(grounded)
    assert len(groups) == 24 and all(len(v) == 2 for v in groups.values())
    result = paired(groups)
    a, b = result["whole"]
    good = (
        a >= 6
        and a - b >= 5
        and result["p"] <= 0.05
        and not result["losses"]
        and counts["candidate"]["unsupported"] <= counts["control"]["unsupported"]
    )
    save(
        report_file,
        {
            "generation_archive": digest(root / "checksums.json"),
            "passed": good,
            "source_pairs": result,
            "arms": counts,
            "ratings_sha256": digest(rating_file),
            "source_decisions": grades_audit,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("blind", "report"))
    parser.add_argument("--streamed", action="store_true")
    parser.add_argument("--variant", choices=("contrastive", "streamed", "elastic"))
    args = parser.parse_args()
    run(args.mode, streamed=args.streamed, variant=args.variant)
