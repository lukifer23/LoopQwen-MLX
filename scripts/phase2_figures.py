"""Draw study figures from registered, complete measurements. Requires the figures group."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from loopqwen.io import read

ROOT = Path("outputs/phase2")
ARCHIVE = Path("docs/measurements/phase2")


def measured(group, arm):
    path = ROOT / group / (arm + ".json")
    if path.exists():
        return read(path)
    return json.loads(
        gzip.decompress((ARCHIVE / group / (arm + ".json.gz")).read_bytes())
    )


def figure(name):
    m = read(Path("experiments") / (name + ".json"))
    dest = ARCHIVE / name
    dest.mkdir(exist_ok=True, parents=True)
    models = [measured(name, run) for run in m["runs"]]
    split = m["splits"][0]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, max(3, len(models) * 0.4 + 1.4)),
        sharey=True,
        layout="constrained",
    )
    positions = np.arange(len(models))
    labels = [
        run.removeprefix("p2_")
        .replace("confirm_", "")
        .replace("screen_", "")
        .replace("a_", "")
        for run in m["runs"]
    ]
    for ax, protocol in zip(axes, ("answer", "trace")):
        x = [r["scores"][split][protocol]["loss"]["nll"] for r in models]
        colors = [
            "#146764" if r["training"]["depth"] > 24 else "#82613a" for r in models
        ]
        ax.barh(positions, x, color=colors)
        ax.set(
            xlabel="Negative log likelihood per target token",
            title=protocol.capitalize() + " prompt",
        )
        ax.set_yticks(positions, labels)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f"{name}   •   {split}")
    fig.savefig(dest / "loss.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    figure(parser.parse_args().experiment)
