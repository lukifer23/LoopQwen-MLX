"""Plot measured sixth-hop accuracy against cumulative curriculum and update time."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("docs/measurements")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True, layout="constrained")
for idx, ax in enumerate(axes):
    result = json.loads((root / f"rsi{idx + 1}" / "decision.json").read_bytes())
    scores = result["summary"]
    for arm, color in [("static", "#206a86"), ("adaptive", "#b65c17")]:
        times = [0.0]
        accuracy = [100 * scores["parent"]["accuracy"][-1]]
        for j in (1, 2):
            name = f"{arm}{j}"
            if name in scores:
                times.append(times[-1] + sum(result["timers"][name].values()))
                accuracy.append(100 * scores[name]["accuracy"][-1])
        ax.plot(times, accuracy, color=color, label=arm.title(), marker="o")
        if idx == 0 and arm == "static":
            extra = json.loads((root / "rsi_time1/decision.json").read_bytes())
            ax.plot(
                [times[-1], times[-1] + extra["extra_spent"]],
                [accuracy[-1], 100 * extra["evaluation"]["accuracy"][-1]],
                color=color,
                linestyle="--",
                marker="s",
                label="Static time control",
            )
    ax.set(
        title=["Seed 719 · two rounds", "Seed 859 · retention stop"][idx],
        xlabel="Selection + diagnosis + update seconds",
        ylim=(55, 103),
        xlim=(-3, 70),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Correct sixth-hop state (%)")
fig.savefig(root / "rsi-quality-cost.png", dpi=160)
