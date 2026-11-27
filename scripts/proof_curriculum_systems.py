"""Place the latest system on the project's measured accuracy-versus-time comparisons."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from loopqwen.io import read


def generate(paper):
    with (paper / "system-comparisons.csv").open(newline="") as f:
        points = []
        for r in csv.DictReader(f):
            points.append(
                {
                    "study": r["study"],
                    "system": r["system"],
                    "endpoint": r["endpoint"],
                    "n": int(r["n"]),
                    "candidate": int(r["candidate"]),
                    "base": int(r["base"]),
                    "seconds_candidate": float(r["seconds_candidate"]),
                    "seconds_base": float(r["seconds_base"]),
                    "first_seconds_candidate": float(r["first_seconds_candidate"])
                    if r["first_seconds_candidate"]
                    else None,
                    "first_seconds_base": float(r["first_seconds_base"])
                    if r["first_seconds_base"]
                    else None,
                    "registered_decision": r["gate"],
                }
            )
    for study, label in (
        ("instruction-interpolation-01", "Interpolated instruction"),
        ("role-binding-01", "Role binding"),
        ("revision-evidence-01", "Revised textual evidence"),
        ("file-evidence-accumulated-01", "Changed file contents"),
        ("cross-view-reading-01", "Cross-view reading"),
        ("proof-curriculum-01", "Proof curriculum"),
    ):
        d = read(Path("research/results") / study / "decision.json")
        if study == "revision-evidence-01":
            a, b = [d["scores"][k] for k in ("candidate", "base")]
            spec = read(Path("research/experiments") / (study + ".json"))
            # Separate the number of enrolled requests from the number which completed.
            n = 2 * spec["pairs"]
            success = "correct"
            wall = "stream_s"
            endpoint = "correct judgments"
        else:
            if study == "proof-curriculum-01":
                a, b = [d["arms"][k]["groups"]["all"] for k in ("candidate", "control")]
            else:
                a, b = [d["groups"][k]["all"] for k in ("candidate", "control")]
            n = a["requests"]
            assert b["requests"] == n
            success = "grounded"
            wall = "wall_s"
            endpoint = "grounded requests"
        points.append(
            {
                "study": study,
                "system": label,
                "endpoint": endpoint,
                "n": n,
                "candidate": a[success],
                "base": b[success],
                "seconds_candidate": a[wall],
                "seconds_base": b[wall],
                "first_seconds_candidate": a["first_s"],
                "first_seconds_base": b["first_s"],
                "registered_decision": "Pass" if d["passed"] else "Miss",
            }
        )
    assert len({p["study"] for p in points}) == len(points)
    for p in points:
        p["accuracy_delta_pp"] = 100 * (p["candidate"] - p["base"]) / p["n"]
        p["wall_speedup"] = p["seconds_base"] / p["seconds_candidate"]
    with (paper / "proof-curriculum-system-inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    fig, ax = plt.subplots(figsize=(7.6, 3.45), layout="constrained")
    for p in points:
        first = p["first_seconds_candidate"]
        if first is None:
            mark, color = "s", "#adb5bd"
        elif first < p["first_seconds_base"]:
            mark, color = "o", "#19826e"
        else:
            mark, color = "^", "#bd7840"
        if p["study"] == "proof-curriculum-01":
            ax.scatter(
                p["accuracy_delta_pp"],
                p["wall_speedup"],
                marker="*",
                s=245,
                c="#392e5d",
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label="Current study",
            )
        else:
            ax.scatter(
                p["accuracy_delta_pp"],
                p["wall_speedup"],
                marker=mark,
                color=color,
                s=47,
                alpha=0.85,
                lw=0.2,
                edgecolors="black",
                zorder=4,
            )
    for marker, color, caption in (
        ("o", "#19826e", "Shorter first wait"),
        ("^", "#bd7840", "Longer or equal first wait"),
        ("s", "#adb5bd", "First wait unavailable"),
    ):
        ax.scatter([], [], marker=marker, color=color, s=40, label=caption)
    ax.axvline(0, color="#56636b", lw=0.85, ls="--")
    ax.axhline(1, color="#56636b", lw=0.85, ls="--")
    ax.set(
        yscale="log",
        xlabel="Change in the study’s reported success rate (percentage points)",
        ylabel="End-to-end speedup vs. Base",
    )
    ax.set_yticks([0.5, 1, 2, 5, 10, 25])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(frameon=False, ncols=2, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(
        paper / "proof-curriculum-system-inventory.svg", metadata={"Date": None}
    )
    plt.close(fig)
    return len(points)
