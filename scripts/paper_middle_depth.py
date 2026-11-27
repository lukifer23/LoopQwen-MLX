"""Build middle-depth tables and plots directly from archived scored trajectories."""

from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import matplotlib
import typst

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv
from scripts.middle_depth_budget import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("paper")


def build():
    scored = []
    for report in sorted((ROOT / "review").glob("*/scored.json")):
        for r in read(report):
            assert digest(ROOT / r["file"]) == r["sha"]
            scored.append(dict(stage=report.parent.name, **r))
    summaries = []
    sources = []
    groups = defaultdict(list)
    for r in scored:
        key = (r["stage"], r["arm"], r["cap"], r["family"])
        groups[key].append(r)
    for (stage, arm, cap, family), rows in groups.items():
        primary = [
            r for r in rows if r["world"] in ("original", "changed", "irrelevant")
        ]
        if not primary:
            continue
        sources_by_id = defaultdict(list)
        for r in primary:
            sources_by_id[r["source"]].append(r)
        rates = lambda deadline: fmean(
            fmean(r["grade"]["correct"] and r["seconds"] <= deadline for r in group)
            for group in sources_by_id.values()
        )
        summaries.append(
            dict(
                stage=stage,
                arm=arm,
                cap=cap,
                family=family,
                total=len(primary),
                correct=sum(r["grade"]["correct"] for r in primary),
                seconds=round(sum(r["seconds"] for r in primary), 2),
                percent=round(100 * rates(float("inf")), 1),
                deadline_percent=round(100 * rates(8), 1),
                median_tps=round(median(r["decode_tps"] for r in primary), 1),
                peak_gib=round(max(r["peak"] for r in primary) / 2**30, 2),
            )
        )
        for source, rs in sources_by_id.items():
            scores = {r["world"]: int(r["grade"]["correct"]) for r in rs}
            sources.append(
                dict(
                    stage=stage,
                    arm=arm,
                    cap=cap,
                    family=family,
                    source=source,
                    **scores,
                    changed_pair=int(
                        bool(scores.get("original") and scores.get("changed"))
                    ),
                    irrelevant_pair=int(
                        bool(scores.get("original") and scores.get("irrelevant"))
                    ),
                )
            )
    records = [
        dict(
            stage=r["stage"],
            arm=r["arm"],
            cap=r["cap"],
            family=r["family"],
            source=r["source"],
            world=r["world"],
            correct=r["grade"]["correct"],
            first_error=r["grade"]["first_error"],
            unsupported=r["grade"]["unsupported"],
            status=r["status"],
            seconds=r["seconds"],
            first_token=r["first_token"],
            first_visible=r["first_visible"],
            time_to_correct=r["time_to_correct"],
            input_tokens=r["input_tokens"],
            generated=r["token_count"],
            decode_tps=r["decode_tps"],
            opening_tps=r["opening_tps"],
            peak_bytes=r["peak"],
            file=r["file"],
            sha256=r["sha"],
        )
        for r in scored
    ]
    write_csv(P / "middle-depth-requests.csv", records)
    write_csv(P / "middle-depth-sources.csv", sources)
    write_csv(P / "middle-depth-families.csv", summaries)
    costs = []
    for f in sorted((ROOT / "charges").glob("*.json")):
        r = read(f)
        if r["status"] == "running":
            continue
        costs.append(
            dict(
                stage=r["stage"],
                job=r["job"],
                status=r["status"],
                seconds=r["seconds"],
                peak_bytes=r["final"]["peak"],
                delta_swap=r["final"]["swap"] - r["initial"]["swap"],
                free_bytes=r["final"]["free"],
                disk_read=(
                    r["final_process_disk_io"]["read"]
                    - r["initial_process_disk_io"]["read"]
                )
                if "final_process_disk_io" in r
                else None,
                disk_written=(
                    r["final_process_disk_io"]["written"]
                    - r["initial_process_disk_io"]["written"]
                )
                if "final_process_disk_io" in r
                else None,
                reason=r.get("reason"),
                receipt=f.name,
            )
        )
    write_csv(P / "middle-depth-costs.csv", costs)
    costsums = {
        k: round(sum(r["seconds"] for r in costs if r["stage"] == k), 2)
        for k in ("profile", "fit", "short", "long")
    }
    plt.rcParams["svg.hashsalt"] = "loopqwen-middle-depth"
    stages = [
        s
        for s in ("initial", "early", "terminal", "simple", "corrected", "deployed")
        if any(r["stage"] == s for r in summaries)
    ]
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(max(7, 2.9 * len(stages)), 3.1),
        sharey=True,
        squeeze=False,
    )
    for ax, stage in zip(axes[0], stages):
        arms = [
            a
            for a in ("base", "ordinary", "loop", "instruction")
            if any(r["stage"] == stage and r["arm"] == a for r in summaries)
        ]
        width = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            values = [
                next(
                    r["percent"]
                    for r in summaries
                    if (r["stage"], r["arm"], r["family"], r["cap"])
                    == (stage, arm, f, 128)
                )
                for f in ("reasoning", "tools", "chat")
            ]
            ax.bar(
                [j + (i - (len(arms) - 1) / 2) * width for j in range(3)],
                values,
                width,
                label=arm,
                color={
                    "base": "#819394",
                    "loop": "#076554",
                    "ordinary": "#e89d49",
                    "instruction": "#5764a1",
                }[arm],
            )
        ax.set_xticks([0, 1, 2], ["Reasoning", "Tools", "Chat"], fontsize=8)
        ax.set_title(stage.capitalize(), fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Supported answers (%)")
    fig.tight_layout()
    fig.savefig(P / "middle-depth-generation.svg", metadata={"Date": None})
    plt.close(fig)
    books = []
    for f in sorted((ROOT / "book").glob("*.gz")) + sorted(
        (ROOT / "documents").glob("*/*/*/*/*.gz")
    ):
        raw = unpack(f)
        tr = raw["trace"]
        t = tr["turns"][0]
        books.append(
            dict(
                file=str(f.relative_to(ROOT)),
                length=len(t["prompt_ids"]),
                status=tr["status"],
                tokens=sum(len(v["token_ids"]) for v in tr["turns"]),
                warm_s=round(tr["seconds"], 2),
                build_s=round(raw.get("cold_build_seconds", 0), 2),
                peak_gib=round(max(v["peak_bytes"] for v in tr["turns"]) / 2**30, 2),
                scratch_gib=round(
                    raw.get("scratch_bytes", t.get("scratch_bytes", 0)) / 2**30, 2
                ),
            )
        )
    if books:
        write_csv(P / "middle-depth-documents.csv", books)
    data = dict(
        summaries=summaries,
        spend=costsums,
        books=books,
        trajectories=len(records),
        source_rows=len(sources),
        archive=read(ROOT / "archive-early.json")["shared_donor_bytes"],
        history=read(ROOT / "history-material.json"),
        decision=read(ROOT / "decision.json")
        if (ROOT / "decision.json").exists()
        else None,
    )
    save(P / "middle-depth-data.json", data)
    typst.compile(
        str(P / "middle-depth.typ"),
        output=str(P / "middle-depth.pdf"),
        root=".",
        timestamp=0,
    )


if __name__ == "__main__":
    build()
