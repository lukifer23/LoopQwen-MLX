"""Count constant answers, opposite answers, actual wheel reads and visible decisions."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.extractive import normalized
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from scripts.visit_residual_reading import PANEL, ROOT


def build():
    human = defaultdict(dict)
    software = {}
    rows = read(PANEL)["rows"]
    table = []
    for arm in ("candidate", "control"):
        counts = Counter()
        for w, case in zip(rows, unpack(ROOT / (arm + ".json.gz")), strict=True):
            (q,) = case["requests"]
            s = q["score"]
            tr = q["trajectory"]
            if w["family"] == "effects":
                text = tr["answer"] or ""
                human[w["cluster"]].setdefault(arm, []).append((w, text, s))
            if w["family"] in ("dependency", "same_version"):
                counts["requests"] += 1
                for key in ("completed", "observed", "range", "verdict", "success"):
                    counts[key] += s[key]
                counts["observed_and_answered"] += s["observed"] and s["completed"]
                counts["observed_answer_range_wrong"] += (
                    s["observed"] and s["completed"] and not s["range"]
                )
        software[arm] = dict(counts)
    summary = {}
    for arm in ("candidate", "control"):
        counts = Counter()
        for key, examples in sorted(human.items()):
            a, b = examples[arm]
            v = [normalized(n) for _, n, _ in (a, b)]
            done = all(x[2]["completed"] for x in (a, b))
            foil = [
                bool(n and n in {normalized(answer) for answer in w["answers"]})
                for (w, _, _), n in zip((b, a), v, strict=True)
            ]
            record = {
                "arm": arm,
                "background": key,
                "both_visible": done,
                "same_answer": bool(done and v[0] == v[1]),
                "one_or_more_exact_foil": any(foil),
                "two_exact_foils": all(foil),
                "both_exact": all(x[2]["success"] for x in (a, b)),
                "first_visible": a[1],
                "second_visible": b[1],
            }
            table.append(record)
            for k in (
                "both_visible",
                "same_answer",
                "one_or_more_exact_foil",
                "two_exact_foils",
                "both_exact",
            ):
                counts[k] += record[k]
        summary[arm] = dict(counts)
    with Path("paper/visit-residual-source-pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    save(
        "docs/measurements/visit-residual-pairs.json",
        {"human": summary, "software": software},
    )
    print(summary, software)


if __name__ == "__main__":
    build()
