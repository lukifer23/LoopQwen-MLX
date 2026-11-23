"""Plot fresh development scores and paired item effects from the frozen 250K record."""

from pathlib import Path
import gzip
import json
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def refinement_figure(archive):
    data = json.loads((archive / "promotion.json").read_text())
    selected = data["selected"]
    splits = [data["splits"][key] for key in ("development", "development_ood")]
    labels = ["Matched dense · K=1", "Gated · K=2", "Same gated weights · K=1"]
    keys = ["dense/trained", selected + "/trained", selected + "/bypass"]
    colors = ["#46586F", "#187C61", "#7A53A0"]
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, (scores, effects) = plt.subplots(1, 2, figsize=(10, 3.95))
        fig.subplots_adjust(left=0.08, right=0.975, bottom=0.22, top=0.74, wspace=0.34)
        fig.suptitle(
            "What did the extra pass contribute?",
            x=0.08,
            y=0.98,
            ha="left",
            fontsize=19,
            fontweight="bold",
        )
        fig.text(
            0.08,
            0.835,
            "250,031 supervised targets per arm   ·   Seed 73   ·   Greedy traced generation",
            color="#46586F",
        )
        for offset, label, key, color in zip((-0.23, 0, 0.23), labels, keys, colors):
            values = [s["metrics"][key] for s in splits]
            bars = scores.bar(
                [i + offset for i in range(2)],
                [r["correct"] for r in values],
                0.2,
                color=color,
                label=label,
            )
            scores.bar_label(
                bars, labels=[str(r["correct"]) for r in values], fontsize=10, padding=2
            )
        scores.set(
            title="Final-answer accuracy",
            ylabel="Problems solved (out of 150)",
            ylim=(0, 72),
            xticks=(0, 1),
            xticklabels=("ID · 150 items", "OOD · 150 items"),
        )
        scores.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.52, -0.12),
            fontsize=8,
            ncol=2,
        )
        effects.axvline(0, color="#637080", linestyle=":", linewidth=1)
        for index, split in enumerate(splits):
            for offset, key, color, mark in (
                (0.13, "against_dense", colors[0], "s"),
                (-0.13, "against_bypass", colors[2], "o"),
            ):
                result = split[key]
                delta = result["delta"] * 100
                lower, upper = [v * 100 for v in result["item_ci95"]]
                effects.errorbar(
                    delta,
                    1 - index + offset,
                    xerr=[[delta - lower], [upper - delta]],
                    color=color,
                    fmt=mark,
                    capsize=4,
                    markersize=6,
                )
        effects.set(
            title="Gated K=2 paired difference",
            xlabel="Percentage points · 95% paired item interval",
            ylim=(-0.45, 1.45),
            yticks=(1, 0),
            yticklabels=("ID", "OOD"),
            xlim=(-4, 9),
        )
        effects.legend(
            handles=[
                Line2D([], [], color=colors[0], marker="s", label="vs matched dense"),
                Line2D([], [], color=colors[2], marker="o", label="vs own K=1"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        for ax in (scores, effects):
            ax.spines["left"].set_color("#B0B7BE")
            ax.spines["bottom"].set_color("#B0B7BE")
        name = archive / "recurrent-contribution.png"
        fig.savefig(name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return name


def memory_figure(archive, selected):
    source = archive / "curves" / ("p3_" + selected + "_250k.json.gz")
    history = json.loads(gzip.decompress(source.read_bytes()))
    blocks = [history[i : i + 32] for i in range(0, len(history), 32)]
    positions, low, high, typical, peak = [], [], [], [], []
    targets = 0
    for block in blocks:
        targets += sum(r["labels"] for r in block)
        positions.append(targets / 1000)
        available = [r["available"] / 2**30 for r in block]
        low.append(min(available))
        high.append(max(available))
        typical.append(median(available))
        peak.append(max(r["peak_bytes"] for r in block) / 2**30)
    boundary = sum(r["labels"] for r in history if not r["checkpointing"]) / 1000
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.fill_between(
        positions, low, high, color="#ccd8e8", label="System available · range"
    )
    ax.plot(
        positions, typical, color="#415975", lw=1.4, label="System available · median"
    )
    ax.plot(positions, peak, color="#14735B", lw=2, label="MLX process peak allocation")
    ax.axvline(boundary, color="#393F45", linestyle=":", lw=1.3)
    ax.annotate(
        "Layer checkpointing enabled",
        (boundary, 0.9),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(
        xlabel="Supervised training tokens (thousands)",
        ylabel="GiB",
        title="Memory during gated training on the 18 GiB M3 Pro",
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.48),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    dest = archive / "training-memory.png"
    fig.savefig(dest, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("docs/measurements/phase3")
    )
    refinement_figure(parser.parse_args().archive)
