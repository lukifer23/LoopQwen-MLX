"""Compare independently replicated answer content and batch-one request time."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import read
from scripts.research_answer_audit import score
from scripts.research_relation_audit import audit

audits = Path("docs/measurements/answer-audits")
results = Path("research/results")
studies = (
    ("learned-exit-01", "Arithmetic · first 32", True),
    ("learned-exit-confirm-01", "Arithmetic · fresh 48", True),
    ("entailment-triplets-01", "Inference · first 48", False),
    ("entailment-triplets-confirm-01", "Inference · fresh 72", False),
)
fig, (quality, time) = plt.subplots(
    1, 2, figsize=(10.4, 3.85), width_ratios=(1.24, 1), layout="constrained"
)
for i, (name, title, arithmetic) in enumerate(studies):
    labels = read(audits / (name + ".json"))
    decision = read(results / name / "decision.json")
    if not decision["passed"]:
        raise ValueError("Every plotted comparison must be qualified and complete")
    if arithmetic:
        reviewed = score(name, labels)
        measured = decision["full"]["all"]
    else:
        reviewed = audit(name, labels)["scores"]["all"]
        measured = decision["scores"]["all"]
    for arm, label, color, offset in (
        ("base", "Untouched Base", "#225A82", 0.18),
        ("candidate", "Trained recurrent variant", "#974320", -0.18),
    ):
        y = i + offset
        n = reviewed["n"]
        correct = reviewed["correct"][arm]
        sec = measured[arm]["seconds"] / n
        quality.barh(
            y,
            100 * correct / n,
            height=0.30,
            color=color,
            label=label if i == 0 else None,
        )
        quality.text(
            100 * correct / n + 1, y, f"{correct}/{n}", va="center", fontsize=9
        )
        time.scatter(sec, y, s=38, color=color, zorder=3)
        time.text(sec * 1.08, y, f"{sec:.2f} s", va="center", fontsize=9)
quality.set(
    yticks=range(len(studies)),
    yticklabels=[row[1] for row in studies],
    xlim=(0, 100),
    xlabel="Correct completed answers after blind review (%)",
)
time.set(
    xscale="log",
    xlim=(0.2, 14),
    yticks=[],
    xlabel="Mean generation time per request (s, log scale)",
)
time.set_xticks([0.3, 1, 3, 10], labels=["0.3", "1", "3", "10"])
for ax in (quality, time):
    ax.set_ylim(len(studies) - 0.55, -0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
fig.legend(
    *quality.get_legend_handles_labels(),
    loc="outside upper center",
    ncol=2,
    frameon=False,
)
fig.savefig("docs/measurements/replicated-reasoning.png", dpi=160)
