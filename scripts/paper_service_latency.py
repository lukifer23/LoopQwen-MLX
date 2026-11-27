"""Empirical user wait distributions from complete cold mixed-request streams."""

import csv
from math import ceil
from statistics import median

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    studies = (
        ("release-mixed-field-01", "Released assistant", 50),
        ("shared-session-01", "Shared reasoning and tools", 120),
        ("shared-session-tokenizer-replay-01", "Unified decoder", 144),
    )
    available = [s for s in studies if (results / s[0] / "checksums.json").is_file()]
    fig, axes = plt.subplots(
        len(available), 2, figsize=(7, 2.6 * len(available)), squeeze=False
    )
    with (paper / "service-latency.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study",
                "arm",
                "timer",
                "requests",
                "min_s",
                "median_s",
                "p90_s",
                "p95_s",
                "max_s",
                "sum_s",
                "faster_than_other_arm",
            ]
        )
        for (study, label, n), ax in zip(available, axes, strict=True):
            result(study)
            root = results / study
            inventory = read(root / "checksums.json")
            arms = {}
            for arm in ("candidate", "control"):
                path = root / (arm + ".json.gz")
                assert digest(path) == inventory[path.name]
                arms[arm] = unpack(path)
            assert len(arms["candidate"]) == n and [
                r["id"] for r in arms["candidate"]
            ] == [r["id"] for r in arms["control"]]
            for x, key, timer in zip(
                ax,
                ("first_token_s", "wall_s"),
                ("First token", "End of request"),
                strict=True,
            ):
                for arm, other, title, color in (
                    ("candidate", "control", label, "#11675D"),
                    ("control", "candidate", "Untouched Base", "#9D5932"),
                ):
                    paired = zip(arms[arm], arms[other], strict=True)
                    faster = sum(a[key] < b[key] for a, b in paired)
                    values = sorted(r[key] for r in arms[arm])
                    assert values[0] > 0
                    writer.writerow(
                        [
                            study,
                            arm,
                            key,
                            n,
                            values[0],
                            median(values),
                            values[ceil(0.90 * n) - 1],
                            values[ceil(0.95 * n) - 1],
                            values[-1],
                            sum(values),
                            faster,
                        ]
                    )
                    x.step(
                        [values[0], *values],
                        [100 * i / n for i in range(n + 1)],
                        where="post",
                        label=title,
                        color=color,
                    )
                x.set_xscale("log")
                x.set_xlabel(timer + " (seconds, log scale)")
                x.set_ylim(0, 102)
            ax[0].set_ylabel("Requests at or below time (%)")
            ax[0].legend(
                loc="upper left", bbox_to_anchor=(0, 1.29), ncol=2, frameon=False
            )
    finish(fig, "service-latency")


def correct_by_deadline(paper, results, result, finish):
    """Count completed successes by observed latency, retaining every enrolled request."""
    studies = [
        ("release-mixed-field-01", "Released assistant"),
        ("shared-session-01", "Shared backbone"),
        ("shared-session-tokenizer-replay-01", "Unified adapter"),
    ]
    fresh_study = studies[-1][0]
    spec = read(results / fresh_study / "registration.json")
    fresh_ids = {
        w["id"]
        for w in read(spec["material"])["rows"]
        if w["family"] in ("math", "judgment", "user_extraction")
    }
    panels = [(*row, None) for row in studies] + [
        (fresh_study, "Fresh human questions", fresh_ids)
    ]
    fig, grid = plt.subplots(2, 2, figsize=(7.0, 4.9), sharey=True)
    with (paper / "correct-by-deadline.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "study",
                "panel",
                "deadline_s",
                "requests",
                "recurrent_completed_correct",
                "base_completed_correct",
                "paired_gains",
                "paired_losses",
                "both_correct",
                "neither_correct",
            ]
        )
        for (study, title, subset), ax in zip(panels, grid.flat, strict=True):
            root = results / study
            result(study)
            checks = read(root / "checksums.json")
            arms = []
            for k in ("candidate", "control"):
                f = root / (k + ".json.gz")
                assert digest(f) == checks[f.name]
                arms.append(
                    [r for r in unpack(f) if subset is None or r["id"] in subset]
                )
            assert [r["id"] for r in arms[0]] == [r["id"] for r in arms[1]]
            n = len(arms[0])
            times = [
                [
                    r["wall_s"]
                    if r["score"]["success"] and r["trajectory"]["status"] == "answered"
                    else float("inf")
                    for r in arm
                ]
                for arm in arms
            ]
            deadlines = sorted(
                {
                    0.125,
                    0.25,
                    0.5,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    *[t for arm in times for t in arm if t != float("inf")],
                }
            )
            values = [[], []]
            for d in deadlines:
                pairs = [(a <= d, b <= d) for a, b in zip(*times, strict=True)]
                g, l = pairs.count((True, False)), pairs.count((False, True))
                both = pairs.count((True, True))
                counts = (both + g, both + l)
                for series, count in zip(values, counts, strict=True):
                    series.append(100 * count / n)
                writer.writerow(
                    [study, title, d, n, *counts, g, l, both, n - both - g - l]
                )
            for series, label, color in zip(
                values, ("Recurrent", "Base"), ("#11675d", "#9d5932"), strict=True
            ):
                ax.step(deadlines, series, where="post", label=label, color=color)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.25, 1, 4, 16], ["0.25", "1", "4", "16"])
            ax.set_title(f"{title} ({n} requests)", fontsize=10)
            ax.set_ylim(0, 80)
    for ax in grid[:, 0]:
        ax.set_ylabel("Completed correctly (%)")
    for ax in grid[-1, :]:
        ax.set_xlabel("Request deadline (seconds)")
    grid[0, 0].legend(frameon=False, fontsize=8)
    finish(fig, "correct-by-deadline")
