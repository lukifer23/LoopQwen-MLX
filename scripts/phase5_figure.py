"""Plot held-out milestones, trained-depth accuracy and probability-rescored trajectories."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figure(root, destination):
    runs = json.loads((root / "training.json").read_bytes())
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), layout="constrained")
    palette = dict(
        probe="#808080",
        middle="#1b7695",
        full="#1d4163",
        recurrent="#c24b26",
        bridge="#8c3e86",
        terminal="#ba7a14",
        multi="#50814b",
        outcome="#937346",
    )
    for arm, run in runs.items():
        metrics = run["metrics"]
        label = {
            "full": "Dense full",
            "middle": "Dense middle",
            "probe": "Frozen probe",
            "recurrent": "Direct recurrent",
            "bridge": "Re-entry bridge",
            "multi": "Dense multi-state",
            "terminal": "Recurrent terminal-only",
            "outcome": "After annealing",
        }.get(arm, arm.title())
        style = dict(color=palette[arm], marker="o", ms=4, lw=1.7)
        axes[1].plot(
            [1, 2, 3], [100 * m["accuracy"] for m in metrics], **style, label=label
        )
        axes[2].plot(
            [1, 2, 3], [100 * m["trajectory"] / m["n"] for m in metrics], **style
        )
        if arm == "outcome":
            continue
        steps = [24, 120, 360, 480]
        acc = []
        for n in steps:
            with gzip.open(root / f"predictions/{arm}-{n}.json.gz", "rt") as f:
                result = json.load(f)
            subset = result["records"][:96]
            acc.append(
                100 * sum(x["pred"][2] == x["gold"][2] for x in subset) / len(subset)
            )
        axes[0].plot([s * 8 for s in steps], acc, **style)
    axes[0].set(
        xlabel="Training map exposures",
        ylabel="Correct (%)",
        title="Three-hop terminal · fixed 96 maps",
    )
    axes[1].set(
        xlabel="Requested hops", title="Terminal state · all 384 maps", xticks=[1, 2, 3]
    )
    axes[2].set(
        xlabel="Requested hops",
        title="Entire trajectory · all 384 maps",
        xticks=[1, 2, 3],
    )
    for ax in axes:
        ax.set_ylim(-2, 103)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.savefig(destination, dpi=155)
    plt.close(fig)
    if (root / "predictions/extra-outcome.json.gz").is_file():
        fig, (deep, reorder) = plt.subplots(
            1, 2, figsize=(8.8, 3.7), layout="constrained"
        )
        for arm, title in (
            ("recurrent", "Installed"),
            ("outcome", "After terminal-only continuation"),
            ("middle", "Dense middle"),
            ("full", "Dense full"),
        ):
            with gzip.open(root / f"predictions/extra-{arm}.json.gz", "rt") as f:
                result = json.load(f)
            deep.plot(
                range(1, 7),
                [100 * m["accuracy"] for m in result["metrics"]],
                marker="o",
                color=palette[arm],
                label=title,
            )
        deep.axvline(3.5, ls="--", color="#666", lw=0.9)
        deep.set(
            title="Held-out maps at six depths",
            xlabel="Requested hops",
            ylabel="Correct (%)",
            xticks=range(1, 7),
        )
        deep.legend(fontsize=7)
        with gzip.open(root / "predictions/diagnostics-recurrent.json.gz", "rt") as f:
            diagnostic = json.load(f)["behavior"]
        for kind, title, style in (
            ("original", "Original order", "-"),
            ("relabel", "Relabeled nodes", "--"),
            ("order", "Shuffled edges", "-"),
        ):
            reorder.plot(
                (1, 2, 3),
                [100 * m["accuracy"] for m in diagnostic[kind]["metrics"]],
                ls=style,
                marker="o",
                label=title,
            )
        reorder.set(
            title="Same 64 maps under rendering changes",
            xlabel="Requested hops",
            xticks=[1, 2, 3],
        )
        reorder.legend(fontsize=8)
        for ax in (deep, reorder):
            ax.set_ylim(-2, 105)
            ax.grid(axis="y", color="#ddd", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(
            destination.with_name(
                destination.stem + "-generalization" + destination.suffix
            ),
            dpi=155,
        )
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/phase5"))
    p.add_argument(
        "--output", type=Path, default=Path("docs/measurements/phase5-pointer.png")
    )
    args = p.parse_args()
    figure(args.archive, args.output)
