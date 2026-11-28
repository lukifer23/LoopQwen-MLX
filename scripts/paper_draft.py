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
from loopqwen.identity import fingerprint
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
                first=round(
                    build
                    + (
                        next(
                            t
                            for pos, t in row["timing"]
                            if pos == len(turn["prompt_ids"])
                        )
                        if "timing" in row
                        else turn["ttft_s"]
                    ),
                    1,
                ),
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
    headroom = use(ROOT / "review/headroom-early/scored.json")
    use(ROOT / "review/headroom-early/reviewer-note.json")
    excerpts = use(ROOT / "excerpt-review/early/scored.json")
    for row in excerpts:
        assert digest(ROOT / row["file"]) == row["sha"]
        inputs[str(ROOT / row["file"])] = row["sha"]
    audit = use(ROOT / "data-audit.json")
    fit_epoch = use(ROOT / "fit-work.json")
    assert [r["steps"] for r in fit_epoch] == [144, 144]
    examples = use("research/sources/middle-depth/conversation-edits.json")
    for r in headroom:
        raw = use(ROOT / r["file"])
        assert digest(ROOT / r["file"]) == r["sha"]
        r["calls"] = [
            {
                k: t[k]
                for k in ("call", "observation", "calls", "observations")
                if k in t
            }
            for t in raw["trace"]["turns"]
            if "call" in t or "calls" in t
        ]
    exits = use(ROOT / "fit-exits-0144.json")
    recovery = use(ROOT / "bounded-fit-operating-point.json")
    completed = use(ROOT / recovery["charge"])
    adjoints = [
        r
        for arm in ("loop", "ordinary")
        for r in use(ROOT / f"adjoint-{arm}.json")["rows"]
    ]
    assert all(r["loss"][0] == r["loss"][1] for r in adjoints)
    prefix = use(ROOT / "prefix-audits/epoch-one-anne.json")
    stopped_book = use(ROOT / "epoch-one-allocator-recovery.json")
    head = use(ROOT / "file-backed-head.json")
    assert (
        prefix["passed"] and head["passed"] and stopped_book["saved_presentations"] == 1
    )
    generation = use(ROOT / "review/epoch-reading/summary.json")
    scored_epoch = use(ROOT / "review/epoch-reading/scored.json")
    for arm in ("loop", "ordinary"):
        use(ROOT / "progress" / f"epoch-reading-{arm}-from-early-{arm}-128.json")
    for r in scored_epoch:
        assert digest(ROOT / r["file"]) == r["sha"]
        inputs[str(ROOT / r["file"])] = r["sha"]
    assert len(scored_epoch) == 124 and len(generation) == 12
    terminal_results = {}
    for stage, count in (
        ("terminal", 248),
        ("quantized", 62),
        ("simple", 62),
        ("corrected", 62),
    ):
        reviewed = use(ROOT / "review" / stage / "scored.json")
        assert len(reviewed) == count
        for r in reviewed:
            assert digest(ROOT / r["file"]) == r["sha"]
            inputs[str(ROOT / r["file"])] = r["sha"]
        terminal_results[stage] = use(ROOT / "review" / stage / "summary.json")
    trained = use(ROOT / "fit-work-0576.json")
    quant = use(ROOT / "quantization/short-decision.json")
    callback = use(ROOT / "quantization/callback-audit.json")
    thoughts = use(ROOT / "training-format-audit.json")
    generated = use(ROOT / "generated-thought-audit.json")
    use(ROOT / "terminal-fit-integrity.json")
    use(ROOT / "pairs/terminal-loop-ordinary-512.json")
    use(ROOT / "installed/terminal-loop-tiles-512.json.gz")
    use(ROOT / "quantization/conversion.json")
    for p in sorted((ROOT / "quantization").glob("*-*-?.json")):
        use(p)

    correction = use(ROOT / "history-decision.json")
    standalone = use(ROOT / "standalone/pair.json")
    use(ROOT / "standalone/restore.json")
    for arm in standalone["order"]:
        path = ROOT / "standalone" / (arm + "-replay.json.gz")
        replay = use(path)
        assert replay["passed"] and replay["pid_absent"]
        assert inputs[str(path)] == standalone["arms"][arm]["trace"]

    opening = use("research/diagnostics/middle-depth/document-openings.json")["anne"][
        "worlds"
    ]["original"]
    anne_pair = []
    for arm, name in (
        ("base", "Untouched Base"),
        ("ordinary", "Trained ordinary"),
        ("loop", "Trained loop"),
    ):
        raw = use(ROOT / f"documents/parent/anne/{arm}/128/original-first.json.gz")
        tr = raw["trace"]
        first = tr["turns"][0]
        assert len(first["prompt_ids"]) == opening["length"]
        assert fingerprint(first["prompt_ids"]) == opening["input"]
        check = use(ROOT / "resident-checks" / (raw["identity"] + ".json"))
        assert check["passed"] and check["maximum_logit_error"] == [0.0, 0.0]
        assert raw["cache_mode"] == "resident" and raw["sites"] == check["sites"]
        assert all(
            site["offset"] == site["retained"]
            for t in tr["turns"]
            for site in t["kv_history"]
        )
        anne_pair.append(
            dict(
                model=name,
                length=opening["length"],
                status=tr["status"],
                generated=sum(len(t["token_ids"]) for t in tr["turns"]),
                build=round(raw["cold_build_seconds"], 1),
                warm=round(tr["seconds"], 1),
                seconds=round(raw["cold_build_seconds"] + tr["seconds"], 1),
                peak=round(max(t["peak_bytes"] for t in tr["turns"]) / 2**30, 2),
                sites=len(raw["sites"]),
                scratch=round(raw["scratch_bytes"] / 2**30, 2),
            )
        )
    anne_scores = {}
    for cap in (128, 512):
        panel = use(ROOT / f"document-review/parent/books/anne-{cap}.json")
        assert len(panel["rows"]) == 6
        scores = {}
        for row in panel["rows"]:
            path = ROOT / row["file"]
            assert digest(path) == row["sha"]
            inputs[str(path)] = row["sha"]
            scores.setdefault(row["arm"], {})[row["world"]] = row["grade"]["correct"]
        assert all(
            set(editions) == {"original", "edited"} for editions in scores.values()
        )
        anne_scores[str(cap)] = scores
    charges = sorted((f, read(f)) for f in (ROOT / "charges").glob("*.json"))
    cutoff = use(ROOT / "charges/48f365a0dbe940268afec646702766ee.json")["dispatch"]
    costs = {}
    for f, r in charges:
        if r["status"] == "running" or r["dispatch"] > cutoff:
            continue
        use(f)
        cost = costs.setdefault(
            r["stage"], {"seconds": 0, "stopped": 0, "stopped_seconds": 0}
        )
        cost["seconds"] += r["seconds"]
        cost["stopped"] += int(r["status"] != "finished")
        if r["status"] != "finished":
            cost["stopped_seconds"] += r["seconds"]
    for v in costs.values():
        v["seconds"] = round(v["seconds"], 1)
        v["stopped_seconds"] = round(v["stopped_seconds"], 1)
    save(
        SNAPSHOT,
        dict(
            title=TITLE,
            authors=AUTHORS,
            headroom=headroom,
            excerpts=excerpts,
            audit=audit,
            fit_epoch=fit_epoch,
            terminal_results=terminal_results,
            correction=correction,
            standalone=standalone,
            final_fit=trained,
            quantization=quant,
            callback_replays=callback["observations"],
            thinking_targets=thoughts["families"],
            generated_thinking=generated["arms"],
            first_epoch_generation=generation,
            exit_objectives=exits["source_weighted"],
            recovery={
                "checks": len(adjoints),
                "relative": max(r["relative"] for r in adjoints),
                "peak_gib": completed["final"]["peak"] / 2**30,
                "prefix_positions": prefix["actual_prefix_tokens"],
                "duplicate_bytes": prefix["avoided_bytes"],
                "saved_presentations": stopped_book["saved_presentations"],
            },
            crate_examples=[v for v in examples if v[0] == "transfer"],
            hardware=hardware,
            historical=historical,
            source_copy=src["scores"],
            early=early,
            costs=costs,
            books=books,
            anne_pair=anne_pair,
            anne_scores=anne_scores,
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

    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    groups = ["reasoning", "tools", "chat", "documents"]
    x = np.arange(len(groups))
    for offset, key, label, color in (
        (-0.18, "loop2_minus_loop1", "Second minus first visit", TEAL),
        (0.18, "loop3_minus_loop2", "Third minus second visit", AMBER),
    ):
        vals = [d["exit_objectives"][g][key]["mean_source_delta"] for g in groups]
        bars = ax.bar(x + offset, vals, width=0.32, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:+.3f}" for v in vals], padding=2, fontsize=8)
    ax.axhline(0, linewidth=0.8, color=INK)
    ax.set_xticks(x, ["Reasoning", "Tools", "Chat", "Documents"])
    ax.set_ylim(-0.061, 0.027)
    ax.set_ylabel("Change in training objective")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    savefig(fig, "exits.svg")

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

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")

    def box(x, width, title, sub, face):
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.0),
                width,
                1.0,
                boxstyle="round,pad=0.07,rounding_size=.12",
                facecolor=face,
                edgecolor="#648078",
                lw=0.85,
            )
        )
        ax.text(
            x + width / 2,
            1.62,
            title,
            fontsize=10.3,
            ha="center",
            va="center",
            weight="bold",
        )
        ax.text(x + width / 2, 1.26, sub, fontsize=8, ha="center", va="center")

    def arrow(a, b, **kwargs):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=11, color=TEAL, lw=1.5, **kwargs
            )
        )

    box(0.13, 1.95, "Prefix", "layers 0–11", "#edf1ef")
    box(3.58, 2.6, "Shared block", "layers 12–15", "#d8eee6")
    box(7.55, 2.16, "Suffix + head", "layers 16–23", "#edf1ef")
    arrow((2.16, 1.50), (3.48, 1.50))
    arrow((6.26, 1.50), (7.44, 1.50))
    ax.text(2.82, 1.82, "anchor", ha="center", fontsize=8)
    ax.text(6.88, 1.82, "chosen exit", ha="center", fontsize=8)
    ax.plot([5.67, 5.67, 4.08], [0.93, 0.43, 0.43], color=TEAL, lw=1.5)
    arrow((4.08, 0.43), (4.08, 0.94))
    ax.text(
        4.9,
        0.07,
        "Mix with the fixed anchor before each revisit",
        fontsize=8.5,
        ha="center",
        color=TEAL,
    )
    savefig(fig, "architecture.svg")
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    display = ("Untouched Base", "Instruction", "Trained ordinary", "Trained loop")
    arms = ("base", "instruction", "ordinary", "loop")
    for i, (arm, name) in enumerate(zip(arms, display, strict=True)):
        row = next(
            r
            for r in d["headroom"]
            if r["arm"] == arm and r["source"] == "tiles" and r["world"] == "original"
        )
        t = row["seconds"]
        ax.hlines(i, 0, t, color="#bdc9c5", linewidth=1.3)
        mark = (
            "o"
            if row["grade"]["correct"]
            else ">"
            if row["status"] == "token_limit"
            else "X"
        )
        ax.scatter(
            t,
            i,
            marker=mark,
            s=60,
            color=TEAL if row["grade"]["correct"] else AMBER,
            zorder=2,
        )
        desc = (
            "correct + executed"
            if row["grade"]["correct"]
            else "token limit"
            if row["status"] == "token_limit"
            else "tool error"
        )
        ax.text(t + 2, i, f"{t:.1f} s · {desc}", va="center", fontsize=8.3)
    ax.axvline(8, color="#536875", ls="--", lw=1)
    ax.text(8.6, 3.58, "8 s deadline", fontsize=8, color="#536875")
    ax.set_yticks(range(4), display)
    ax.set_xlim(0, 106)
    ax.set_ylim(3.9, -0.45)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Elapsed seconds on the same invoice request")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    savefig(fig, "headroom.svg")


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
        assert 5 <= len(pages) <= 12 and all(len(s) > 700 for s in pages)
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
