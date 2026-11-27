"""Summarize paired source-level differences for both scoring rules and real project pairs."""

from collections import defaultdict

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.linked_rollin import exact_sign_flip
from scripts.visit_residual_reading import ROOT, PANEL, PLAN, RATINGS


def stats(values):
    return {
        "units": len(values),
        "gains": sum(x > 0 for x in values),
        "losses": sum(x < 0 for x in values),
        "ties": sum(x == 0 for x in values),
        "net": sum(values),
        "two_sided_sign_flip": exact_sign_flip(values),
    }


def build():
    rows = read(PANEL)["rows"]
    selected = set(read(PLAN)["selected"])
    grades = read(RATINGS)
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for arm in ("candidate", "control"):
        for w, r in zip(rows, unpack(ROOT / (arm + ".json.gz")), strict=True):
            q = r["requests"][0]
            if w["family"] in ("effects", "dependency", "same_version"):
                by[w["family"]][w["cluster"]][arm].append(int(q["score"]["success"]))
            if w["family"] != "effects" or w["id"] not in selected:
                continue
            ob = {
                "request": w["questions"][0],
                "reference": {"published_answers": w["answers"]},
                "visible_answer": q["trajectory"]["answer"] or "",
            }
            judged = grades[fingerprint(ob)[:20]]
            by["masked_human"][w["cluster"]][arm].append(
                int(judged["verdict"] == "supported")
            )
    out = {}
    for family, clusters in by.items():
        assert all(
            len(group[arm]) == 2
            for group in clusters.values()
            for arm in ("candidate", "control")
        )
        dif = [sum(v["candidate"]) - sum(v["control"]) for v in clusters.values()]
        pairs = [
            int(all(v["candidate"])) - int(all(v["control"])) for v in clusters.values()
        ]
        out[family] = {
            "questions_within_source": stats(dif),
            "whole_source": stats(pairs),
        }
    assert (
        out["effects"]["questions_within_source"]["two_sided_sign_flip"]
        == read(ROOT / "decision.json")["paired_p"]
    )
    save("docs/measurements/visit-residual-source-statistics.json", out)
    print(out)


if __name__ == "__main__":
    build()
