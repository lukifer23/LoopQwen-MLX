"""Recount real per-visit training margins, protected displacements and loss movement."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save

ROOT = Path("research/results/followup-protected-01")
PAPER = Path("paper")


def build(root=ROOT):
    measures = read(root / "optimizer-directions.json")
    views = unpack(root / "training-views.json.gz")
    f = read(root / "fit.json")
    assert len(measures) == f["updates"]
    windows = []
    details = []
    for start in range(0, len(measures), 100):
        part = measures[start : start + 100]
        kinds = defaultdict(list)
        for step in part:
            for m in step["margins"]:
                kind = views[m["view"]]["kind"]
                kinds[kind].append(m)
                details.append(
                    {
                        "step": step["step"],
                        "view_id": views[m["view"]]["id"],
                        "kind": kind,
                        **m,
                    }
                )
        means = {}
        for kind, group in kinds.items():
            first, second = np.array([[m["first"], m["second"]] for m in group]).T
            means[kind] = {
                "draws": len(group),
                "first": float(first.mean()),
                "second": float(second.mean()),
                "gained": int(np.sum((first <= 0) & (second > 0))),
                "lost": int(np.sum((first > 0) & (second <= 0))),
                "mean_delta": float((second - first).mean()),
            }
        windows.append(
            {
                "through": part[-1]["step"],
                "proposed": len(part),
                "nonzero": sum(r["applied_step_l2"] > 0 for r in part),
                "projected": sum(r["projected"] for r in part),
                "backtracked": sum(0 < r["factor"] < 1 for r in part),
                "rolled_back": sum(r["factor"] == 0 for r in part),
                "maximum_reference_increase": max(
                    r["after"] - r["before"] for r in part
                ),
                "samples": means,
            }
        )
    summary = {
        "windows": windows,
        "sampled_foils": len(details),
        "unique_contrasted_views": len({r["view"] for r in details}),
        "draws_by_kind": dict(Counter(r["kind"] for r in details)),
        "proposed_l2_median": float(
            np.median([r["proposed_step_l2"] for r in measures])
        ),
        "applied_l2_median": float(np.median([r["applied_step_l2"] for r in measures])),
    }
    save("docs/measurements/followup-learning.json", summary)
    with (PAPER / "followup-training-margins.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, details[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)
    plt.rcParams.update(
        {
            "svg.hashsalt": "followup-protected-01",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    xs = [g["through"] for g in windows]
    for kind, title, color in (
        ("human_first", "First question", "#39729a"),
        ("human_followup", "Follow-up question", "#84579b"),
        ("action", "Archive choice", "#286956"),
        ("interpret", "Compatibility answer", "#b35f39"),
    ):
        a.plot(
            xs,
            [g["samples"].get(kind, {}).get("mean_delta", np.nan) for g in windows],
            marker=".",
            label=title,
            color=color,
        )
    a.axhline(0, color=".5", lw=0.6)
    a.set(
        xlabel="Optimizer step, in groups of 100",
        ylabel="Second minus first logit margin",
    )
    a.legend(frameon=False, fontsize=7)
    for key, title, color in (
        ("nonzero", "Changed weights", "#286956"),
        ("projected", "Projected", "#39729a"),
        ("backtracked", "Backtracked", "#b35f39"),
        ("rolled_back", "Rolled back", "#84579b"),
    ):
        b.plot(xs, [g[key] for g in windows], marker=".", label=title, color=color)
    b.set(
        xlabel="Optimizer step, in groups of 100",
        ylabel="Updates per group",
        ylim=(-3, 105),
    )
    b.legend(frameon=False, fontsize=7)
    fig.savefig(PAPER / "followup-learning.svg", metadata={"Date": None})
    plt.close(fig)
    print(
        len(details), "actual pre-update contrasts in", len(measures), "optimizer steps"
    )


if __name__ == "__main__":
    build()
