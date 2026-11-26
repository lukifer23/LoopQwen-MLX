"""Recount a shipped assistant's cold and mixed-mode field workloads."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.mixed_field import outcome


def make(paper, results, result, finish):
    name = "release-mixed-field-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    checks = read(root / "checksums.json")
    data = {}
    for side in ("candidate", "control"):
        source = root / (side + ".json.gz")
        if digest(source) != checks[source.name]:
            raise ValueError("Field transcript changed")
        data[side] = unpack(source)
    if outcome(data, read(root / "registration.json")) != decision:
        raise ValueError("Field outcomes differ from the paired trajectories")
    labels = {
        "reading": "Reading",
        "arithmetic": "Arithmetic",
        "database": "SQLite",
        "package": "Packages",
        "file": "Files",
        "all": "Pooled",
    }
    header = [
        "family",
        "side",
        "n",
        "correct",
        "completed",
        "tokens",
        "wall_s",
        "first_token_s",
        "peak_bytes",
    ]
    rows = []
    for group in labels:
        for side in data:
            cell = decision["by_family"][group][side]
            rows.append(
                [
                    group,
                    side,
                    *[
                        cell[k]
                        for k in (
                            "n",
                            "correct",
                            "completed",
                            "tokens",
                            "seconds",
                            "first_token_seconds",
                            "peak",
                        )
                    ],
                ]
            )
    with (paper / "mixed-deployment.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.65), sharey=True)
    items = list(labels)
    for i, key in enumerate(items):
        pooled = decision["by_family"][key]

        def values(s):
            return (
                100 * s["correct"] / s["n"],
                s["seconds"] / s["n"],
                1000 * s["first_token_seconds"] / s["n"],
                s["peak"] / 2**30,
            )

        candidate, control = map(values, (pooled["candidate"], pooled["control"]))
        for j, (ax, a, b) in enumerate(zip(axes, candidate, control, strict=True)):
            ax.plot([a, b], [i, i], lw=1, color="#aeb8b4")
            ax.scatter(
                a, i - 0.045, s=22, c="#11675d", label="Recurrent" if not i else None
            )
            ax.scatter(
                b,
                i + 0.045,
                s=20,
                c="#9d5932",
                marker="s",
                label="Base" if not i else None,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0].set_yticks(range(len(items)), [labels[k] for k in items], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-4, 104)
    for ax, title in zip(
        axes,
        ("Correct (%)", "Seconds / query", "Mean TTFT (ms)", "Peak (GiB)"),
        strict=True,
    ):
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[1].legend(
        loc="lower left", bbox_to_anchor=(-0.3, 1.16), frameon=False, ncol=2, fontsize=8
    )
    finish(fig, "mixed-deployment")
