"""Publish the stopped explanation qualification from every training-only generation."""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition

P = Path("paper")


def build():
    summary = read("docs/measurements/reasoned-evidence-teacher.json")
    root = Path("research/results/reasoned-evidence-01")
    registration = read(root / "registration.json")
    assert (
        read(root / "decision.json")["status"] == "interrupted"
        and not (root / "fit.json").exists()
    )
    with (P / "reasoned-evidence-teacher.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(r["status"] for r in rows) == summary["stops"]
    labels = ("true", "false", "uncertain")
    colors = ("#237365", "#bc783d", "#7c8baf", "#baabb6", "#c6ccca")
    statuses = [
        "Accepted",
        "Reached output cap",
        "Explanation length",
        "Different verdict",
        "Missing or invalid premise citation",
    ]
    legends = ["Accepted", "Token limit", "Length", "Wrong verdict", "Citation"]
    plt.rcParams.update(
        {
            "svg.hashsalt": "reasoned-evidence-01",
            "font.family": "DejaVu Sans",
            "font.size": 9,
        }
    )
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.6, 3.9), layout="constrained", height_ratios=[1, 1.07]
    )
    for i, label in enumerate(labels):
        counts = summary["by_label"][label]
        start = 0
        for j, (status, color) in enumerate(zip(statuses, colors, strict=True)):
            size = counts.get(status, 0)
            ax.barh(
                i, size, left=start, color=color, label=legends[j] if i == 0 else None
            )
            if size >= 5:
                ax.text(
                    start + size / 2,
                    i,
                    str(size),
                    ha="center",
                    va="center",
                    color="white" if j < 2 else "#172527",
                )
            start += size
    ax.set(
        yticks=range(3),
        yticklabels=[k.capitalize() for k in labels],
        xlim=(0, 64),
        xlabel="Generated explanations per published verdict",
    )
    ax.invert_yaxis()
    ax.axvline(
        registration["teacher_min_per_label"], color="#222", linestyle="--", lw=0.85
    )
    ax.legend(
        ncols=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.36),
        fontsize=8,
    )
    for label, color in zip(labels, ("#237365", "#af6c35", "#5f7299"), strict=True):
        lengths = sorted(int(r["tokens"]) for r in rows if r["human"] == label)
        bx.step(
            lengths,
            [(i + 1) / len(lengths) for i in range(len(lengths))],
            where="post",
            color=color,
            label=label.capitalize(),
        )
    bx.set(
        xlabel="Tokens emitted before termination or truncation",
        ylabel="Cumulative share",
        xlim=(0, 180),
        ylim=(0, 1.05),
    )
    bx.axvline(registration["teacher_cap"], color="#222", ls="--", lw=0.85)
    bx.legend(loc="upper left", frameon=False, ncols=3, fontsize=8)
    for chart in (ax, bx):
        chart.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "reasoned-evidence-qualification.svg", metadata={"Date": None})
    plt.close(fig)
    formal = read("docs/measurements/reasoned-evidence-source-oracle.json")
    review = read("research/diagnostics/reasoned-evidence-language-review.json")
    accepted = summary["stops"]["Accepted"]
    report = {
        "headline": f"The teacher qualified {accepted} of {len(rows)} candidate explanations ({100 * accepted / len(rows):.1f}%). The registered minimum was {registration['teacher_min_total']}.",
        "counts": ", ".join(
            f"{summary['by_label'][label]['Accepted']} {label}" for label in labels
        ),
        "resource": f"{summary['tokens']:,} tokens in {summary['seconds']:.2f} seconds, with {summary['peak_bytes'] / 2**30:.2f} GiB peak allocation",
        "terminal": f"{summary['terminal_verdict']} generations ended with a recognizable verdict and {summary['terminal_correct']} matched the publisher.",
        "oracle": f"{formal['agreements']}/{formal['answers']} formal labels agreed. Prior English review flagged {len(review['flagged'])} of the {formal['worlds']} stories.",
    }
    save(P / "reasoned-evidence-paper.json", report)
    edition(P, "reasoned-evidence")


if __name__ == "__main__":
    build()
