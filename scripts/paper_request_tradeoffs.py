"""Score strict paired user-level gains jointly with measured latency and memory."""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read

COSTS = ("first_token_s", "wall_s", "peak")


def make(paper, results, result, finish):
    study = "shared-session-tokenizer-replay-01"
    root = results / study
    result(study)
    material = read(read(root / "registration.json")["material"])["rows"]
    a, b = [unpack(root / (s + ".json.gz")) for s in ("candidate", "control")]
    ordered = defaultdict(list)
    with (paper / "paired-request-tradeoffs.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "family",
                "recurrent_correct",
                "base_correct",
                *[s + "_" + k for k in COSTS for s in ("recurrent", "base")],
                "cost_ratios_recurrent_over_base_max",
                "unique_win_all_costs",
                "matched_success_all_costs",
                "both_wrong_all_costs",
                "base_unique_all_costs",
            ]
        )
        for w, x, y in zip(material, a, b, strict=True):
            assert w["id"] == x["id"] == y["id"]
            correct, base = x["score"]["success"], y["score"]["success"]
            assert min(v[k] for v in (x, y) for k in COSTS) > 0
            ratio = max(x[k] / y[k] for k in COSTS)
            better = ratio <= 1
            loss = not correct and base and all(y[k] <= x[k] for k in COSTS)
            writer.writerow(
                [
                    x["id"],
                    w["family"],
                    int(correct),
                    int(base),
                    *[v[k] for k in COSTS for v in (x, y)],
                    ratio,
                    int(correct and not base and better),
                    int(correct and base and better),
                    int(not correct and not base and better),
                    int(loss),
                ]
            )
            ordered[w["family"]].append((correct, base, ratio))
    fig, ax = plt.subplots(figsize=(7.05, 3.25))
    colors = ["#11675d", "#88bb9e", "#536a89", "#dedbd4"]
    names = [
        "Gained answer, all costs ≤ Base",
        "Both correct, all costs ≤ Base",
        "Both wrong, all costs ≤ Base",
        "Remaining requests",
    ]
    all_rows = [row for rs in ordered.values() for row in rs]
    groups = {"All": all_rows, **ordered}
    with (paper / "paired-request-tradeoffs-summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "requests",
                "maximum_cost_ratio",
                "new_correct_within_limit",
                "both_correct_within_limit",
                "both_wrong_within_limit",
                "remaining",
                "all_quality_gains",
                "all_quality_losses",
            ]
        )
        for i, (family, rows) in enumerate(groups.items()):
            for limit in [0.95, 0.99, 1.0]:
                counts = [
                    sum(c and not b and r <= limit for c, b, r in rows),
                    sum(c and b and r <= limit for c, b, r in rows),
                    sum(not c and not b and r <= limit for c, b, r in rows),
                ]
                counts.append(len(rows) - sum(counts))
                writer.writerow(
                    [
                        family,
                        len(rows),
                        limit,
                        *counts,
                        sum(c and not b for c, b, r in rows),
                        sum(b and not c for c, b, r in rows),
                    ]
                )
                if limit != 1:
                    continue
                left = 0
                for n, color, label in zip(counts, colors, names, strict=True):
                    ax.barh(
                        i,
                        100 * n / len(rows),
                        left=left,
                        height=0.7,
                        color=color,
                        label=label if i == 0 else None,
                    )
                    left += 100 * n / len(rows)
    ax.set_yticks(
        range(len(groups)),
        [
            {
                "user_extraction": "Source extraction",
                "judgment": "Yes/no judgment",
                "math": "Fresh arithmetic",
                "arithmetic": "Field arithmetic",
                "public": "Public functions",
                "files": "Local files",
                "file": "Local files",
                "package": "Package versions",
                "sql": "SQLite",
            }.get(k, k.capitalize())
            for k in groups
        ],
    )
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Percent of paired requests")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.2),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    finish(fig, "paired-request-tradeoffs")
