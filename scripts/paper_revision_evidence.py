"""Produce article-paired tables, figures and the consolidated publication from native records."""

import csv
import json
from pathlib import Path

import matplotlib

from scripts.assemble_evidence_paper import edition

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from scripts.revision_material import NAME, PANEL

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
OUT = ROOT / "research/results" / NAME


def main():
    rows = read(PANEL)
    d = read(OUT / "decision.json")
    notes = {
        r["case"]: r
        for r in read(ROOT / "research/diagnostics/revision-source-reading.json")[
            "articles"
        ]
    }
    revisions = {
        str(r["revision"]): r
        for r in read(ROOT / "docs/measurements/revision-wikipedia-sources.json")[
            "articles"
        ]
    }
    arms = {a: unpack(OUT / (a + ".json.gz")) for a in ("candidate", "base")}
    per_case = {}
    details = []
    for i, r in enumerate(rows):
        line = {
            k: r[k]
            for k in ("id", "page", "case", "revision", "claim", "evidence", "label")
        }
        note = notes[r["case"]]
        line.update(
            source_reading=note["rating"],
            source_reason=note["reason"],
            wiki_diff=revisions[r["revision"]]["url"],
        )
        for a, records in arms.items():
            x = records[i]
            assert x["id"] == r["id"]
            line.update(
                {a + "_" + k: v for k, v in x["score"].items() if k != "visible"}
            )
            line[a + "_visible"] = json.dumps(x["score"]["visible"], ensure_ascii=False)
            line[a + "_request_seconds"] = x["request_s"] + x["load_s"]
            line[a + "_tokens"] = len(x["token_ids"])
            per_case.setdefault(r["case"], {}).setdefault(a, []).append(x["score"])
        details.append(line)
    with (P / "revision-evidence-requests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=details[0])
        w.writeheader()
        w.writerows(details)
    sides = ("candidate", "base")
    c, b = (d["scores"][a] for a in sides)
    direct = [k for k, n in notes.items() if n["rating"] == "direct"]

    def ns(arm, keys):
        return sum(all(r["correct"] for r in per_case[k][arm]) for k in keys)

    def same(arm):
        return sum(
            a["answer"] == b["answer"] and a["answer"] is not None
            for k in per_case
            for a, b in [per_case[k][arm]]
        )

    def fmt(x):
        return f"{x:.2f}"

    cert = read(OUT / "compact-entailment.json")
    groups = [
        "Both edits\ncorrect",
        "Direct pairs\nboth correct",
        "Supported\ncorrect",
        "Refuted\ncorrect",
    ]
    colors = ("#17645b", "#8e4f2c")
    plt.rcParams.update(
        {
            "svg.hashsalt": "loopqwen-revision-evidence",
            "font.size": 9,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(7.45, 2.65), layout="constrained")
    for i, (a, color, label) in enumerate(
        zip(sides, colors, ("Compact K2", "Untouched Base"))
    ):
        x = d["scores"][a]
        vals = [x["pairs"], ns(a, direct), x["supports"], x["refutes"]]
        limits = [len(per_case), len(direct), len(per_case), len(per_case)]
        bars = ax.bar(
            [j + (i - 0.5) * 0.35 for j in range(4)],
            vals,
            width=0.34,
            color=color,
            label=label,
        )
        ax.bar_label(
            bars,
            labels=[f"{v}/{n}" for v, n in zip(vals, limits)],
            padding=2,
            fontsize=8,
        )
    ax.set(
        xticks=range(4), xticklabels=groups, ylabel="Articles or requests", ylim=(0, 28)
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.19), frameon=False)
    fig.savefig(P / "revision-evidence.svg", metadata={"Date": None})
    plt.close(fig)
    costs = [
        ("Complete opposite pairs", c["pairs"], b["pairs"]),
        (
            "Both correct on direct-source subset",
            ns("candidate", direct),
            ns("base", direct),
        ),
        ("Correct individual judgments", c["correct"], b["correct"]),
        ("Completed thinking turns", c["complete"], b["complete"]),
        ("Same parsed label across both edits", same("candidate"), same("base")),
        ("Cold stream, seconds", fmt(c["stream_s"]), fmt(b["stream_s"])),
        (
            "Cumulative time to first token, seconds",
            fmt(c["first_s"]),
            fmt(b["first_s"]),
        ),
        (
            "Peak MLX allocation, GiB",
            fmt(c["peak_bytes"] / 2**30),
            fmt(b["peak_bytes"] / 2**30),
        ),
        ("Native input tokens/s", fmt(c["prefill_tps"]), fmt(b["prefill_tps"])),
        ("Native output tokens/s", fmt(c["decode_tps"]), fmt(b["decode_tps"])),
    ]
    typ = [
        '#table(columns:(2.7fr,1fr,1fr),align:(left,right,right),stroke:none,fill:(x,y)=>if y==0 {rgb("#d8ebe6")} else if calc.odd(y) {rgb("#f1f6f5")}, [Measure],[Compact K2],[Base],'
    ]
    typ.extend(f"[{k}],[{v}],[{w}]," for k, v, w in costs)
    (P / "revision-evidence-costs.typ").write_text("\n".join(typ) + ")\n")
    summary = f"The loop correctly answered both snapshots for *{c['pairs']}/24* distinct Wikipedia articles, compared with *{b['pairs']}/24* for ordinary Base. It won {d['wins']} article pairs and lost {d['losses']} (two-sided exact paired p = {d['paired_p']:.4g}). Across all 48 individual requests it gave {c['correct']} correct judgments and Base gave {b['correct']}. The two cold half-streams took {fmt(c['stream_s'])} and {fmt(b['stream_s'])} seconds in total, respectively."
    (P / "revision-evidence-abstract.typ").write_text(summary + "\n")
    gate_words = {
        "opposing_pairs": "opposite-pair accuracy and significance",
        "supports": "supported-claim accuracy",
        "refutes": "refuted-claim accuracy",
        "completion": "completion",
        "stream_s": "stream time",
        "first_s": "first-token time",
        "peak_bytes": "peak allocation",
        "prefill_tps": "input throughput",
        "decode_tps": "output throughput",
    }
    missed = [gate_words[k] for k, v in d["gates"].items() if not v]
    gate_text = (
        "All registered gates passed."
        if d["passed"]
        else "The registered comparison missed "
        + ", ".join(missed)
        + ". This closes the proposed depth comparison, field promotion and long-context extension for this policy."
    )
    audited = read(ROOT / "docs/measurements/revision-answer-reading.json")
    (P / "revision-evidence-interpretation.typ").write_text(
        f"On the {len(direct)} articles judged directly distinguishable before reading model output, complete paired accuracy was {ns('candidate', direct)}/{len(direct)} versus {ns('base', direct)}/{len(direct)}. Across all articles the two snapshots received the same parsed conclusion in {same('candidate')} pairs for the loop and {same('base')} for Base. {gate_text} A separate blinded review of {audited['distinct_reviewed']} distinct completed noncanonical replies gives {audited['candidate']['items']} versus {audited['base']['items']} correct individual conclusions, and {audited['candidate']['pairs']} versus {audited['base']['pairs']} complete article pairs.\n"
    )
    (P / "revision-evidence-conversion.typ").write_text(
        f"The conversion fused {len(cert['fused'])} low-rank projections. The learned gate was {cert['learned_gate']:.5f}. The last-position relative RMS logit difference on an independent prose sentence was {cert['relative_logit_rms']:.4f}, below the registered 0.06 limit. Building and writing the checkpoint took {cert['seconds']:.2f} s to its manifest and peaked at {cert['peak_bytes'] / 2**30:.2f} GiB. The raw eight-bit tensor file contains {read(ROOT / 'docs/measurements/compact-revision-runtime.json')['weight_bytes']:,} bytes. These are one-time conversion costs. Serving measurements include the production loads and their weight checks.\n"
    )
    edition(P, "revision-evidence")


if __name__ == "__main__":
    main()
