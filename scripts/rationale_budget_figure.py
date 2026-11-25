"""Plot first-turn performance at greedy prefixes of the archived fresh cohort."""

from bisect import bisect_right
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("research/results/rationale-transfer-01")
result = json.loads((root / "decision.json").read_bytes())
fig, (quality, timing) = plt.subplots(
    1, 2, figsize=(10, 3.8), width_ratios=(1.6, 1), layout="constrained"
)
for arm, label, color, offset in (
    ("base", "Untouched Base K1", "#225A82", 0.16),
    ("candidate", "Rationale-trained K2", "#9C431F", -0.16),
):
    with gzip.open(root / (arm + ".json.gz")) as stream:
        rows = json.load(stream)["records"]
    finish = sorted(len(row["token_ids"]) for row in rows if row["correct"])
    caps = [0, *sorted(set(finish)), 1024]
    quality.step(
        caps,
        [bisect_right(finish, x) for x in caps],
        where="post",
        label=label,
        color=color,
        linewidth=2.3,
    )
    for y, budget in enumerate(("prefix_256", "full")):
        seconds = result[budget]["all"][arm]["seconds"]
        timing.barh(y + offset, seconds, height=0.29, color=color)
        timing.text(
            seconds + 5, y + offset, f"{seconds:.0f} s", va="center", fontsize=9
        )
quality.axvline(256, linestyle=":", color="#555555", linewidth=1)
quality.set(
    title="Correct first turns across 48 questions",
    xlabel="Output allowance (tokens)",
    ylabel="Questions answered correctly",
    xlim=(0, 1024),
    ylim=(0, 48),
    xticks=(0, 256, 512, 768, 1024),
)
quality.legend(loc="upper right", frameon=False)
timing.set(
    title="Measured time for the full cohort",
    xlabel="Generation time (seconds)",
    yticks=(0, 1),
    yticklabels=("256 tokens", "1,024 tokens"),
    xlim=(0, 445),
)
timing.invert_yaxis()
for axis in (quality, timing):
    axis.spines[["top", "right"]].set_visible(False)
fig.savefig("docs/measurements/rationale-budget.png", dpi=180)
