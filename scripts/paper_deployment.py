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


def transport(paper, results, result, finish):
    name = "transport-lossless-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    sizes = read(root / "sizes.json")
    if sum(sizes.values()) != d["raw_bytes"] or read(root / "original.json") != read(
        root / "restored.json"
    ):
        raise ValueError("Transport figure cannot verify the installed payload")
    stats = (
        ("Base text weights", d["raw_base_weight_bytes"]),
        ("Installed assistant", d["raw_bytes"]),
        ("Ordinary ZIP", d["ordinary_zip_bytes"]),
        ("Paired encoding", d["bytes"]),
    )
    times = (
        ("Ordinary ZIP pack", d["ordinary_zip_seconds"]),
        ("Paired pack", d["pack_seconds"]),
        ("Paired restore", d["restore_seconds"]),
    )
    with (paper / "transport.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["measure", "unit", "value"])
        w.writerows((label, "bytes", value) for label, value in stats)
        w.writerows((label, "seconds", value) for label, value in times)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.2))
    for ax, cells, unit, scale in zip(
        axes, (stats, times), ("GiB", "Seconds"), (2**30, 1), strict=True
    ):
        for i, (label, value) in enumerate(cells):
            v = value / scale
            ax.barh(
                i, v, color="#11675d" if "aired" in label else "#8a9b98", height=0.53
            )
            ax.text(
                v,
                i,
                f"  {v:.2f}" if unit == "GiB" else f"  {v:.1f}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(range(len(cells)), [label for label, _ in cells], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(v for _, v in cells) / scale * 1.27)
        ax.set_xlabel(unit)
        ax.spines[["right", "top"]].set_visible(False)
    finish(fig, "transport")
