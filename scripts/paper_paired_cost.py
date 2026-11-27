"""Compare success and three measured service costs jointly for every complete dialogue."""

import csv

from loopqwen.evaluation.mechanism import unpack
from loopqwen.research.native_function_study import summarize


def metrics(row):
    if "requests" in row:
        d = summarize([row])
        return d["strict"], d["wall_s"], d["first_s"], d["peak"]
    return (
        int(row["grade"]["exact"]),
        row["cold_seconds"],
        row["cold_first"],
        row["peak_bytes"],
    )


def write(paper, results):
    studies = {
        "native-function-dialogue-01": "Published functions",
        "observation-review-01": "Receipt review",
        "question-contrast-01": "Question contrasts",
        "document-frequency-reader-01": "Whole documents",
    }
    with (
        (paper / "paired-costs.csv").open("w", newline="") as raw,
        (paper / "paired-costs-summary.csv").open("w", newline="") as summary,
    ):
        observations, totals = csv.writer(raw), csv.writer(summary)
        fields = ["strict", "wall_seconds", "first_seconds", "peak_bytes"]
        observations.writerow(
            [
                "study",
                "id",
                "family",
                *("loop_" + k for k in fields),
                *("base_" + k for k in fields),
                "all_costs_lower",
                "quality",
            ]
        )
        totals.writerow(
            [
                "study",
                "n",
                "gains",
                "gains_all_three",
                "losses",
                "losses_all_three",
                "both_correct",
                "both_correct_loop_cheaper",
                "both_correct_base_cheaper",
                "both_failed",
            ]
        )
        for name, title in studies.items():
            root = results / name
            if not (root / "checksums.json").is_file():
                continue
            rows = [unpack(root / (s + ".json.gz")) for s in ("candidate", "control")]
            counts = dict.fromkeys(
                (
                    "gain",
                    "gain_loop",
                    "loss",
                    "loss_base",
                    "correct",
                    "correct_loop",
                    "correct_base",
                    "failed",
                ),
                0,
            )
            for a, b in zip(*rows, strict=True):
                assert a["id"] == b["id"] and a.get("family") == b.get("family")
                loop, base = metrics(a), metrics(b)
                costs = list(zip(loop[1:], base[1:], strict=True))
                owner = (
                    "loop"
                    if all(x < y for x, y in costs)
                    else "base"
                    if all(y < x for x, y in costs)
                    else "mixed_or_equal"
                )
                if loop[0] != base[0]:
                    quality = "gain" if loop[0] else "loss"
                else:
                    quality = "correct" if loop[0] else "failed"
                counts[quality] += 1
                key = quality + "_" + owner
                if key in counts:
                    counts[key] += 1
                observations.writerow(
                    [
                        name,
                        a["id"],
                        a.get("family", "document"),
                        *loop,
                        *base,
                        owner,
                        quality,
                    ]
                )
            n = len(rows[0])
            assert sum(counts[k] for k in ("gain", "loss", "correct", "failed")) == n
            totals.writerow([title, n, *counts.values()])
