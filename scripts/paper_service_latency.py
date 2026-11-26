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
