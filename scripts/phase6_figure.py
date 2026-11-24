"""Render matched order, extrapolation, independent latency and memory measurements."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def plot(root, dest):
    data = load(root / "results.json")["arms"]
    labels = {
        "canonical": "Canonical recurrent",
        "randomized": "Randomized recurrent",
        "middle": "Dense middle",
        "full": "Dense full",
        "heads": "Dense three-head",
        "outcome": "Terminal continuation",
    }
    colors = {
        "canonical": "#c97914",
        "randomized": "#007780",
        "middle": "#62626a",
        "full": "#4b72ae",
        "heads": "#975995",
        "outcome": "#579f4b",
    }
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.4), layout="constrained")
    a, b, c, d, e, f = ax.ravel()
    for name, run in data.items():
        style = {
            "color": colors[name],
            "marker": "o",
            "lw": 1.7,
            "ms": 4,
            "label": labels[name],
        }
        points = []
        for ev in run["evaluations"]:
            cohort = [
                r
                for r in load(root / ev["path"])["records"]
                if r["condition"] == "random0"
            ][:96]
            hit = sum(
                max(range(8), key=r["probabilities"][2].__getitem__) == r["gold"][2]
                for r in cohort
            )
            points.append((ev["step"] * 8, 100 * hit / len(cohort)))
        if name != "outcome":
            a.plot(*zip(*points), **style)
        scores = ev["scores"]["conditions"]
        b.plot([1, 2, 3], [100 * v for v in scores["mean_random"]["accuracy"]], **style)
        c.plot(
            [0, 1],
            [100 * scores[k]["trajectory"][-1] for k in ("canonical", "mean_random")],
            **style,
        )
        if run.get("deep"):
            sums = load(root / name / "depth-summary.json")["groups"]
            ys = []
            for hop in range(1, 7):
                groups = [s for k, s in sums.items() if k.startswith(f"random0/{hop}/")]
                ys.append(
                    100
                    * sum(v["correct"] for v in groups)
                    / sum(v["n"] for v in groups)
                )
            d.plot(range(1, 7), ys, **style)
        ms = next(
            r["median_ms"]
            for r in load(root / run["latency"])["results"]
            if r["batch"] == 1 and r["depth"] == 3
        )
        e.scatter(
            ms,
            scores["mean_random"]["accuracy"][2] * 100,
            color=colors[name],
            marker="o",
        )
        f.scatter(run["maps_s"], run["peak_gib"], color=colors[name], marker="o")
    a.set(
        title="Third-hop learning curve",
        xlabel="Training map exposures",
        ylabel="Correct (%)",
    )
    b.set(
        title="Eight shuffled orders",
        xlabel="Requested hop",
        ylabel="Correct (%)",
        xticks=[1, 2, 3],
    )
    c.set(
        title="Complete three-hop paths",
        xticks=[0, 1],
        xticklabels=["Canonical", "Shuffled mean"],
        ylabel="Correct (%)",
    )
    d.set(
        title="Extrapolation on one shuffle",
        xlabel="Shared block applications",
        ylabel="Correct (%)",
        xticks=range(1, 7),
    )
    e.set(
        title="Three-hop inference",
        xlabel="Batch-one latency (ms)",
        ylabel="Shuffled accuracy (%)",
    )
    f.set(
        title="Single-worker training",
        xlabel="Maps / update second",
        ylabel="Peak MLX allocation (GiB)",
    )
    for axis in ax.ravel():
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in [a, b, c, d, e]:
        axis.set_ylim(-3, 104)
    fig.legend(
        *b.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(dest, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/phase6"))
    p.add_argument(
        "--output", type=Path, default=Path("docs/measurements/phase6-orders.png")
    )
    args = p.parse_args()
    plot(args.archive, args.output)
