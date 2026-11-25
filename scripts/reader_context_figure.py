"""Plot verified register answers and latency from the two frozen memory archives."""

import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path("research/results")
panels = (
    ("attention-memory-01", slice(0, 4)),
    ("attention-memory-01", slice(4, 8)),
    ("attention-memory-128k", slice(0, 2)),
)
fig, (quality, timing) = plt.subplots(
    1, 2, figsize=(9.1, 3.6), width_ratios=(1, 1.28), layout="constrained"
)
for arm, label, color, offset in (
    ("base", "Untouched Base K1", "#225A82", -0.17),
    ("candidate", "Learned reader K2", "#974320", 0.17),
):
    for i, (name, selection) in enumerate(panels):
        if (
            json.loads((results / name / "decision.json").read_bytes())["status"]
            != "completed"
        ):
            raise ValueError("The figure requires both completed primary comparisons")
        with gzip.open(results / name / (arm + ".json.gz")) as f:
            rows = json.load(f)["records"][selection]
        count = sum(r["correct"] for r in rows)
        y = i + offset
        quality.barh(
            y,
            count / len(rows),
            height=0.27,
            color=color,
            label=label if i == 0 else None,
        )
        quality.text(
            count / len(rows) + 0.025,
            y,
            f"{count}/{len(rows)}",
            va="center",
            fontsize=9,
        )
        sec = mean(r["elapsed_s"] for r in rows)
        timing.scatter(sec, y, s=45, color=color, zorder=3)
        timing.text(sec * 1.13, y, f"{sec:.1f} s", va="center", fontsize=9)
for ax in (quality, timing):
    ax.set(yticks=(0, 1, 2), yticklabels=("2K", "16K", "128K"), ylim=(2.52, -0.58))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e4e6", linewidth=0.7)
quality.set(
    xlim=(0, 1.18),
    xticks=(0, 0.5, 1),
    xticklabels=("0%", "50%", "100%"),
    title="Completed correct answers",
)
timing.set_xscale("log")
timing.set(
    xlim=(1, 680),
    xticks=(1, 10, 100),
    xticklabels=("1", "10", "100"),
    xlabel="Mean generation seconds per request (log scale)",
    title="Wall time including full prefill",
)
quality.legend(
    loc="lower center", bbox_to_anchor=(0.5, -0.31), frameon=False, ncol=2, fontsize=8
)
fig.suptitle(
    "Full-history register lookup on Apple M3 Pro", fontsize=13, fontweight="bold"
)
fig.savefig("docs/measurements/reader-context.png", dpi=180)
