"""Recount declared compatibility with a separately recorded post-primary reading of the answers."""

import csv
from collections import defaultdict
from pathlib import Path

from loopqwen.io import read, save
from loopqwen.research.linked_rollin import exact_sign_flip

P = Path("paper")
DOC = Path("docs/measurements")


def build():
    review = read(DOC / "followup-wheel-review.json")["reviewed_exceptions"]
    used = set()
    with (P / "followup-requests.csv").open(newline="") as f:
        items = [r for r in csv.DictReader(f) if r["family"] == "archive"]
    assert len(items) == 24
    rows = []
    for r in items:
        row = {"id": r["id"], "package": r["cluster"]}
        for a in ("candidate", "control"):
            original = r[a + "_success"] == "True"
            observed = (
                r[a + "_current_read"] == "True" and r[a + "_status"] == "answered"
            )
            assert not original or observed
            allowed = original
            if observed and not original:
                audit = review[r["id"]]
                allowed = audit[a]
                used.add(r["id"])
                row["review_reason"] = audit["reason"]
            row[a + "_registered"] = original
            row[a + "_reviewed"] = allowed
        rows.append(row)
    assert used == review.keys()
    totals = {}
    for method in ("registered", "reviewed"):
        clusters = defaultdict(list)
        for r in rows:
            clusters[r["package"]].append(
                [r[a + "_" + method] for a in ("candidate", "control")]
            )
        whole = [
            (all(a for a, b in v), all(b for a, b in v)) for v in clusters.values()
        ]
        assert len(whole) == 12 and all(len(v) == 2 for v in clusters.values())
        delta = [int(a) - int(b) for a, b in whole]
        totals[method] = {
            "requests": [
                sum(r[a + "_" + method] for r in rows) for a in ("candidate", "control")
            ],
            "projects": [sum(v[j] for v in whole) for j in (0, 1)],
            "gains": delta.count(1),
            "losses": delta.count(-1),
            "descriptive_exact_p": exact_sign_flip(delta),
        }
    save(DOC / "followup-wheel-sensitivity.json", totals)
    with (P / "followup-wheel-sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            [
                "id",
                "package",
                "candidate_registered",
                "control_registered",
                "candidate_reviewed",
                "control_reviewed",
                "review_reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return totals
