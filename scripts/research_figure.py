"""Render the two measured generalization boundaries from archived development scores."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("research/results")
depth = json.loads((root / "trace-depth-01/decision.json").read_text())["scores"]
graph = json.loads((root / "functional-zero-01/decision.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
for k, label, color in (
    ("permutation", "Restricted permutations", "#175783"),
    ("functional", "Arbitrary successors", "#a04227"),
):
    ax[0].plot(
        range(1, 7),
        [100 * v / graph[k]["n"] for v in graph[k]["correct"]],
        marker="o",
        color=color,
        label=label,
    )
ax[0].set(
    xlabel="Shared-block applications",
    ylabel="Correct state (%)",
    xticks=range(1, 7),
    ylim=(-4, 105),
    title="One checkpoint, 192 maps per distribution",
)
ax[0].legend(frameon=False, fontsize=8)
for name, label, dx, dy, color in (
    ("base", "Base K=1", 4, 6, "#666666"),
    ("dense", "Trained K=1", 6, -17, "#175783"),
    ("bypass", "K=3-trained at K=1", 6, 5, "#696039"),
    ("recurrent", "Trained K=3", -68, -17, "#a04227"),
):
    s = depth[name]
    x, y = s["median_ms"], 100 * s["verified"] / s["n"]
    ax[1].scatter(x, y, color=color, s=43)
    ax[1].annotate(
        label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8
    )
ax[1].set(
    xlabel="Median batch-one request (ms)",
    ylabel="Fully verified trace (%)",
    ylim=(-4, 105),
    xlim=(130, 477),
    title="Same three-hop questions, 96 fresh maps",
)
for a in ax:
    a.spines[["right", "top"]].set_visible(False)
    a.grid(axis="y", color="#e5e5e5", linewidth=0.65)
fig.savefig("docs/measurements/research-generalization.png", dpi=180)
