"""Synthesize selected direct comparisons and expose the complete program inventory."""

import csv
from pathlib import Path

import matplotlib
import numpy as np
import typst
from pypdf import PdfReader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import digest, read, save
from loopqwen.evaluation.mechanism import unpack
from scripts.research_scope import inventory

P = Path("paper")
R = Path("research/results")


def build():
    inventory()
    scope = read(P / "research-scope.json")
    rows = []

    def add(study, domain, comparison, n, metrics, arms, correct, elapsed):
        for role, arm in zip(("loop", "comparison"), arms, strict=True):
            m = metrics[arm]
            rows.append(
                dict(
                    study=study,
                    domain=domain,
                    comparator=comparison,
                    role=role,
                    n=n,
                    successful=m[correct],
                    seconds=m[elapsed],
                    inventory_sha256=digest(R / study / "checksums.json"),
                )
            )

    for name, domain, comparison in (
        ("native-induction-anchored-01", "Induction", "Untouched Base"),
        ("native-induction-anchored-matched-01", "Induction", "Trained K=1"),
        ("native-induction-anchored-replication-01", "Fresh induction", "Trained K=1"),
        ("native-induction-anchored-seed2-01", "New fit seed", "Trained K=1"),
        ("native-induction-class-swap-01", "Changed class", "Trained K=1"),
    ):
        data = read(R / name / "decision.json")["scores"]
        add(
            name,
            domain,
            comparison,
            64,
            data,
            ("recurrent", "base" if "base" in data else "dense"),
            "correct",
            "seconds",
        )
    for name, comparison in (
        ("tool-dialogue-typed-01", "Untouched Base"),
        ("tool-dialogue-typed-depth-01", "Trained K=1"),
    ):
        d = read(R / name / "decision.json")
        add(
            name,
            "Public executions",
            comparison,
            d["items"],
            d["metrics"],
            ("candidate", "control"),
            "success",
            "seconds",
        )
    for name, domain in (
        ("valid-return-01", "Alternate tool results"),
        ("linked-rollin-02", "Learner-generated histories"),
        ("followup-protected-01", "Human and wheel follow-ups"),
        ("visit-residual-01", "Second-visit readout"),
    ):
        if name == "visit-residual-01" and not (P / "visit-residual-paper.json").exists():
            continue
        if name.startswith("followup") and not (P / "followup-paper.json").exists():
            continue
        d = read(R / name / "decision.json")
        data = {
            k: (v["groups"] if "groups" in v else v)["all"]
            for k, v in d["arms"].items()
        }
        add(
            name,
            domain,
            "Untouched Base",
            data["candidate"]["requests"],
            data,
            ("candidate", "control"),
            "grounded",
            "wall_s",
        )
    name = "register-packed-direct-full-01"
    d = read(R / name / "decision.json")
    add(
        name,
        "Long changed fact",
        "Untouched Base",
        2,
        d["metrics"],
        ("direct_pages", "base"),
        "correct",
        "seconds",
    )
    name = "book-conversation-field-01"
    add(
        name,
        "Book requests",
        "Untouched Base",
        6,
        read(R / name / "decision.json")["metrics"],
        ("candidate", "control"),
        "success",
        "wall",
    )
    name = "locked-context-01"
    add(
        name,
        "Complete lock revisions",
        "Untouched Base",
        2,
        read(R / name / "decision.json")["totals"],
        ("recurrent", "base"),
        "grounded",
        "wall_s",
    )
    pairs = list(zip(rows[::2], rows[1::2], strict=True))
    with (P / "overview-comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "svg.hashsalt": "loopqwen-evidence-overview",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.85),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    offsets = np.arange(5)
    names = ("Primary", "New stories", "New seed", "Changed class", "Public tools")
    selection = [pairs[k] for k in (1, 2, 3, 4, 6)]
    for side, (label, color) in enumerate(
        (("Loop", "#176657"), ("Trained one pass", "#786094"))
    ):
        a.barh(
            offsets + (side - 0.5) * 0.34,
            [x[side]["successful"] / x[side]["n"] for x in selection],
            height=0.31,
            color=color,
            label=label,
        )
    a.set(
        yticks=offsets,
        yticklabels=names,
        xlim=(0, 1.06),
        xlabel="Fraction of original items correct",
    )
    a.invert_yaxis()
    a.legend(
        fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1, -0.45), ncol=2
    )
    for j, (x, y) in enumerate(selection):
        b.plot(x["seconds"] / y["seconds"], j, "o", color="#176657", ms=5)
    b.axvline(1, color=".5", lw=1)
    b.set(
        yticks=offsets,
        yticklabels=[],
        xlabel="Loop / one-pass elapsed time",
        xlim=(0.97, 1.07),
        ylim=(4.5, -0.5),
    )
    fig.savefig(P / "overview-depth.svg", metadata={"Date": None})
    plt.close(fig)
    studies = [
        (x["study"], x["domain"])
        for x, _ in pairs
        if x["study"].startswith(
            ("valid-return", "linked-rollin", "followup-protected", "visit-residual")
        )
    ]
    fig, axes = plt.subplots(
        1, len(studies), figsize=(7.15, 2.4), sharey=True, layout="constrained"
    )
    deadline_rows = []
    for axis, (study, domain) in zip(axes, studies, strict=True):
        for arm, color, label in (
            ("candidate", "#176657", "Loop"),
            ("control", "#786094", "Base"),
        ):
            reqs = [
                q for r in unpack(R / study / (arm + ".json.gz")) for q in r["requests"]
            ]
            successes = sorted(q["wall_s"] for q in reqs if q["score"]["success"])
            maximum = max(q["wall_s"] for q in reqs)
            times = [0, *successes, max(30, maximum)]
            quantities = [0, *range(1, len(successes) + 1), len(successes)]
            axis.step(
                times,
                np.array(quantities) / len(reqs),
                where="post",
                color=color,
                label=label,
            )
            deadline_rows.extend(
                {
                    "study": study,
                    "arm": arm,
                    "seconds": sec,
                    "successes": n,
                    "requests": len(reqs),
                }
                for sec, n in zip(times, quantities, strict=True)
            )
        axis.set(
            title=domain,
            xlabel="Deadline (seconds)",
            xscale="symlog",
            xlim=(0, 30),
            ylim=(0, 1),
        )
        axis.set_xticks([0, 1, 3, 10, 30], labels=["0", "1", "3", "10", "30"])
    axes[0].set_ylabel("Fraction correct by deadline")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.savefig(P / "overview-delivery.svg", metadata={"Date": None})
    plt.close(fig)
    with (P / "overview-deadlines.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, deadline_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(deadline_rows)
    summary = []
    for a, b in pairs:
        assert (
            a["study"] == b["study"]
            and 0 <= a["successful"] <= a["n"]
            and 0 <= b["successful"] <= b["n"]
        )
        summary.append(
            [
                a["domain"],
                a["comparator"],
                str(a["n"]),
                str(a["successful"]),
                str(b["successful"]),
                f"{a['seconds']:.2f}",
                f"{b['seconds']:.2f}",
            ]
        )
    save(P / "overview-data.json", {"rows": summary, "scope": scope})
    doc = typst.compile(str(P / "overview.typ"), timestamp=0)
    (P / "evidence-overview.pdf").write_bytes(doc)
    pages = PdfReader(P / "evidence-overview.pdf").pages
    assert len(pages) == 4 and all(len(p.extract_text()) > 550 for p in pages)
    print(len(rows) // 2, "direct comparisons;", len(pages), "pages")


if __name__ == "__main__":
    build()
