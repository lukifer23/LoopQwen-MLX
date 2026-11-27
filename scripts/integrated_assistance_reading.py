"""Judge all genuinely open assistance and SQL replies before inspecting arm totals."""

import argparse
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, save

PLAN = Path("research/diagnostics/integrated-assistance-reading.json")
ROOT = Path("research/results/integrated-assistance-01")
DOC = Path("docs/measurements")
MASK = DOC / "integrated-assistance-blinded.json"
RATINGS = DOC / "integrated-assistance-blinded-ratings.json"


def run(mode):
    rows = read("research/data/integrated-assistance-panel.json")["rows"]
    selected = [w for w in rows if w["family"] in ("open", "database")]
    criteria = {
        "open": "Judge the complete reply against the user's request. A strong answer fulfills its explicit parts, is accurate on material claims and is usable in context. Credit valid alternatives to the published reply. For creative requests judge adherence, specificity and whether the requested artifact was delivered. For incomplete information credit a targeted clarifying question or a useful conditional answer. Distinguish unsupported specificity, significant omissions and clear factual errors. Mark each reply fully usable, partial, wrong or absent, record a brief source-based rationale, and compare user preference allowing a tie.",
        "database": "Judge the visible reply against the human request, its recorded result and the original human response. Accept alternative wording or numeric formatting when the requested fact is clear. Record extra unsupported claims or incorrect qualifications. The independent query suite decides tool grounding.",
        "release": "Require at least six first-place preferences out of twelve open requests and more wins than losses, no fewer fully usable replies and no more material unsupported claims than Base. Separately require no fewer semantically supported database answers. A failed automatic gate closes promotion even if readers prefer the candidate.",
    }
    plan = {
        "selected": [w["id"] for w in selected],
        "criteria": criteria,
        "arms_blinded": True,
        "method": "Order replies independently per question using a deterministic digest of their visible text. Identical replies necessarily tie. Expose no lengths, token streams, timings, generation status, automatic evaluation or policy identifiers. Seal all judgments in git before producing any mapping.",
    }
    assert len(selected) == 36
    if mode == "plan":
        save(PLAN, plan)
        return
    assert plan == read(PLAN)
    runs = {
        a: {
            r["id"]: r["requests"][0]["trajectory"]["answer"] or ""
            for r in unpack(ROOT / (a + ".json.gz"))
        }
        for a in ("candidate", "control")
    }
    mask = {}
    order = {}
    for w in selected:
        key = w["id"]
        replies = sorted(
            {r[key] for r in runs.values()},
            key=lambda t: fingerprint([197264, key, t]),
        )
        mask[key] = {
            "request": w["questions"][0],
            "reference": w.get("human_reply", w.get("human_answer")),
            "replies": replies,
            "family": w["family"],
        }
        if w["family"] == "database":
            mask[key]["expected_value"] = w["expected_value"]
        order[key] = {a: replies.index(v[key]) for a, v in runs.items()}
    if mode == "blind":
        if MASK.exists():
            assert read(MASK) == mask
        else:
            save(MASK, mask)
        return
    assert read(MASK) == mask
    judgments = read(RATINGS)
    assert judgments.keys() == mask.keys()
    totals = {a: {f: Counter() for f in ("open", "database")} for a in runs}
    for key, j in judgments.items():
        labels = j["replies"]
        assert len(labels) == len(mask[key]["replies"]) and j["reason"].strip()
        assert j["preferred"] in ["tie", *range(len(labels))]
        if len(labels) == 1:
            assert j["preferred"] == "tie"
        for arm in totals:
            idx = order[key][arm]
            grade = labels[idx]
            assert (
                grade["rating"] in ("fully usable", "partial", "wrong", "absent")
                and type(grade["unsupported"]) is bool
            )
            rec = totals[arm][mask[key]["family"]]
            rec[grade["rating"]] += 1
            rec["unsupported"] += grade["unsupported"]
            rec["win"] += j["preferred"] == idx
            rec["tie"] += j["preferred"] == "tie"
    a, b = (totals[k] for k in ("candidate", "control"))
    passed = (
        a["open"]["win"] >= max(6, b["open"]["win"] + 1)
        and a["open"]["fully usable"] >= b["open"]["fully usable"]
        and a["open"]["unsupported"] <= b["open"]["unsupported"]
        and a["database"]["fully usable"] >= b["database"]["fully usable"]
    )
    save(
        DOC / "integrated-assistance-reading.json",
        {"arms": totals, "reading_gate": passed},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "blind", "report"))
    run(p.parse_args().mode)
