"""Render the review manuscript from a frozen cut of native evidence."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
import tempfile

import matplotlib
import numpy as np
from pypdf import PdfReader
import typst

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("paper")
SNAPSHOT = P / "paper-draft-data.json"
AUTHORS = ["lukifer23", "GPT 6.2 Astra"]
TITLE = "When Does Shared Depth Earn Its Cost?"
TEAL = "#087365"
AMBER = "#a35520"
INK = "#22313e"


def snapshot():
    inputs = {}

    def use(path):
        path = Path(path)
        inputs[str(path)] = digest(path)
        return read(path) if path.suffix == ".json" else unpack(path)

    hardware = use(ROOT / "hardware.json")
    comparisons = P / "overview-comparisons.csv"
    inputs[str(comparisons)] = digest(comparisons)
    with comparisons.open() as stream:
        hist = list(csv.DictReader(stream))
    historical = []
    for study, label in (
        ("native-induction-anchored-matched-01", "Rule induction"),
        ("native-induction-anchored-confirmation-01", "Fresh stories"),
        ("native-induction-anchored-seed2-01", "Second fit seed"),
        ("native-induction-class-swap-01", "Changed class"),
    ):
        rs = [r for r in hist if r["study"] == study]
        if not rs and label == "Fresh stories":
            rs = [r for r in hist if r["domain"] == "Fresh induction"]
        assert len(rs) == 2
        historical.append(
            dict(
                label=label,
                loop=int(rs[0]["successful"]),
                ordinary=int(rs[1]["successful"]),
                n=int(rs[0]["n"]),
                loop_s=float(rs[0]["seconds"]),
                ordinary_s=float(rs[1]["seconds"]),
                study=rs[0]["study"],
            )
        )
    public = [
        r
        for r in hist
        if r["domain"] == "Public executions" and r["comparator"] == "Trained K=1"
    ]
    historical.append(
        dict(
            label="Public tools",
            loop=int(public[0]["successful"]),
            ordinary=int(public[1]["successful"]),
            n=int(public[0]["n"]),
            loop_s=float(public[0]["seconds"]),
            ordinary_s=float(public[1]["seconds"]),
            study=public[0]["study"],
        )
    )
    src = use("docs/measurements/source-copy-supported-delivery.json")
    early = use(ROOT / "review/early/summary.json")
    scored = use(ROOT / "review/early/scored.json")
    pair = use(ROOT / "pairs/early-loop-ordinary-128.json")
    both = [
        use(ROOT / "book/initial-loop-original-128.json.gz"),
        use(ROOT / "documents/initial/book/base/128/original-first.json.gz"),
    ]
    books = []
    for name, row in zip(("Initial two-pass", "Untouched Base"), both, strict=True):
        tr = row["trace"]
        turn = tr["turns"][0]
        build = row.get("prefix_seconds", 0)
        books.append(
            dict(
                name=name,
                length=len(turn["prompt_ids"]),
                output=len(turn["token_ids"]),
                status=tr["status"],
                first=round(build + (next(t for pos,t in row["timing"] if pos==len(turn["prompt_ids"])) if "timing" in row else turn["ttft_s"]), 1),
                total=round(build + tr["seconds"], 1),
                sites=len(turn["kv_history"]),
                peak=round(turn["peak_bytes"] / 2**30, 2),
            )
        )
    profile = use(ROOT / "profile-base.json")
    data = use("research/sources/middle-depth/inventory.json")
    resident = use(ROOT / "resident-qualification.json")
    restart = use(ROOT / "restart-loop.json")
    repair = use("docs/measurements/receipt-repair-fit-streamed/closure.json")
    use(ROOT / "prepared-restart-loop.json")
    use(ROOT / "prepared-restart-ordinary.json")
    use(ROOT / "history-material.json")
    use(ROOT / "release-regression.json")
    use(ROOT / "donor-decision.json")
    use(ROOT / "sparse-parity.json")
    # Closed charges up to and including exact resident/callback qualification
    # delimit a reproducible engineering cost cut during a longer running fit.
    charges = [(f, read(f)) for f in (ROOT / "charges").glob("*.json")]
    cutoff = max(
        r["dispatch"]
        for f, r in charges
        if r["job"] == "full-prefix resident-vs-staged output parity"
        and r["status"] == "finished"
    )
    costs = {}
    for f, r in charges:
        if r["status"] == "running" or r["dispatch"] > cutoff:
            continue
        use(f)
        cost = costs.setdefault(r["stage"], {"seconds": 0, "stopped": 0})
        cost["seconds"] += r["seconds"]
        cost["stopped"] += int(r["status"] != "finished")
    for v in costs.values():
        v["seconds"] = round(v["seconds"], 1)
    save(
        SNAPSHOT,
        dict(
            title=TITLE,
            authors=AUTHORS,
            hardware=hardware,
            historical=historical,
            source_copy=src["scores"],
            early=early,
            costs=costs,
            books=books,
            resident=resident["rows"],
            profile=profile["rows"],
            params=profile["trainable"],
            train=data["quotas"],
            restart_max=restart["maximum"],
            repair=repair,
            first_cap=128,
            changes=[
                dict(
                    source=r["source"],
                    world=r["world"],
                    delta=r["gain"],
                    loop=r["left"]["grade"],
                    ordinary=r["right"]["grade"],
                )
                for r in pair["trajectories"]
                if r["gain"]
            ],
            early_times={
                a: round(sum(r["seconds"] for r in early if r["arm"] == a), 2)
                for a in ("ordinary", "loop")
            },
            curves={
                a: [
                    sum(
                        r["grade"]["correct"] and r["seconds"] <= t
                        for r in scored
                        if r["arm"] == a
                        and r["world"] in ("original", "changed", "irrelevant")
                    )
                    for t in (0.5, 1, 2, 4, 8)
                ]
                for a in ("ordinary", "loop")
            },
            input_files=inputs,
        ),
    )


def figures(d, tmp):
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.labelsize": 10,
            "svg.hashsalt": "loopqwen-paper-draft",
        }
    )

    def savefig(fig, name):
        fig.tight_layout()
        fig.savefig(tmp / name, metadata={"Date": None}, bbox_inches="tight")
        plt.close(fig)

    hist = d["historical"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for i, r in enumerate(hist):
        vals = [100 * r[k] / r["n"] for k in ("ordinary", "loop")]
        ax.plot(vals, [i, i], color="#a2aaa7", lw=2, zorder=1)
        ax.scatter(
            vals[0],
            i,
            marker="s",
            s=58,
            facecolor="white",
            edgecolor=AMBER,
            linewidth=1.8,
            zorder=2,
            label="Trained one-pass" if i == 0 else None,
        )
        ax.scatter(
            vals[1],
            i,
            s=48,
            color=TEAL,
            zorder=3,
            label="Trained loop" if i == 0 else None,
        )
    ax.set_yticks(range(len(hist)), [r["label"] + f"  (n={r['n']})" for r in hist])
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Successful episodes (%)")
    ax.grid(axis="x", alpha=0.17)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    savefig(fig, "history.svg")
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.08, 1]}
    )
    for idx, arm in enumerate(("ordinary", "loop")):
        color = (AMBER, TEAL)[idx]
        label = ("Ordinary depth", "Two-pass loop")[idx]
        rs = {r["family"]: r for r in d["early"] if r["arm"] == arm}
        vals = [rs[f]["seconds"] for f in ("reasoning", "tools", "chat")]
        y = np.arange(3) + (idx - 0.5) * 0.32
        bars = a.barh(y, vals, height=0.31, label=label, color=color)
        a.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ts = [0, 0.5, 1, 2, 4, 8]
        b.step(
            ts,
            [0, *d["curves"][arm]],
            where="post",
            lw=2.4,
            color=color,
            linestyle=("-", "--")[idx],
            label=label,
        )
    a.set_yticks(range(3), ["Reasoning", "Tools", "Chat"])
    a.invert_yaxis()
    a.set_xlim(0, 19)
    a.set_xlabel("Seconds for nine requests")
    b.set(
        xlim=(0, 8),
        ylim=(-0.3, 6),
        xlabel="Per-request deadline (seconds)",
        ylabel="Correct completions out of 27",
    )
    b.set_xticks([0, 2, 4, 6, 8])
    b.set_yticks([0, 2, 4, 6])
    b.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (a, b):
        ax.grid(axis="x", alpha=0.15)
    savefig(fig, "time.svg")


def render():
    d = read(SNAPSHOT)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        figures(d, tmp)
        (tmp / "data.json").write_text(json.dumps(d))
        (tmp / "paper.typ").write_bytes((P / "paper-draft.typ").read_bytes())
        pdf = typst.compile(str(tmp / "paper.typ"), root=str(tmp), timestamp=0)
        check = tmp / "review.pdf"
        check.write_bytes(pdf)
        reader = PdfReader(check)
        pages = [p.extract_text() for p in reader.pages]
        assert 5 <= len(pages) <= 9 and all(len(s) > 700 for s in pages)
        assert all(author in pages[0] for author in AUTHORS)
        (P / "paper-draft.pdf").write_bytes(pdf)
        print(len(pages), "pages", len(pdf), "bytes", digest(P / "paper-draft.pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    if args.refresh:
        snapshot()
    render()
