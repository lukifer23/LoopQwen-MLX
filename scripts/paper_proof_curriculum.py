"""Produce paper tables, figures and synthesis exclusively from sealed MLX trajectories."""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition
from scripts.proof_curriculum_reading import STUDY, extract, report

P = Path("paper")
GREEN = "#126c5d"
BLUE = "#6573a3"
ARMS = ("candidate", "control")
NAMES = ("Trained loop", "Untouched Base")


def build():
    rows, _, _ = extract()
    d = read(STUDY / "decision.json")
    arms = d["arms"]
    by = report(rows)
    assert by["all"]["net"] == d["net"] and by["all"]["paired_p"] == d["exact_story_p"]
    fit = read(STUDY / "fit.json")
    exposure = read("docs/measurements/proof-curriculum-training-exposure.json")
    blind = read("docs/measurements/proof-curriculum-blind-reading.json")
    hist = read(STUDY / "optimizer-steps.json")
    panel = read(read(STUDY / "registration.json")["material"])
    a, b = [arms[k] for k in ARMS]
    groups = ("inline", "file", "wheel", "retention")
    plt.rcParams.update(
        {
            "svg.hashsalt": "proof-curriculum-01",
            "font.family": "DejaVu Sans",
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.65, 2.65), layout="constrained")
    xx = np.arange(4)
    for arm, name, col, dx in zip(
        ARMS, NAMES, (GREEN, BLUE), (-0.19, 0.19), strict=True
    ):
        vals = [arms[arm]["groups"][g] for g in groups]
        ax.bar(
            xx + dx,
            [v["grounded"] / v["requests"] for v in vals],
            0.36,
            label=name,
            color=col,
        )
        for i, v in enumerate(vals):
            ax.text(
                i + dx,
                v["grounded"] / v["requests"] + 0.025,
                f"{v['grounded']}/{v['requests']}",
                ha="center",
                fontsize=8,
            )
    ax.set(
        xticks=xx,
        xticklabels=["Inline logic", "File logic", "Compatibility", "Table retention"],
        ylabel="Grounded success",
        ylim=(0, 1.18),
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    ax.legend(ncols=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "proof-curriculum-accuracy.svg", metadata={"Date": None})
    plt.close(fig)

    cases = defaultdict(list)
    for r in rows:
        if r["family"] in ("inline", "file"):
            cases[r["cluster"]].append(r)
    fig, ax = plt.subplots(figsize=(7.65, 2.75), layout="constrained")
    keys = sorted(cases)
    for i, k in enumerate(keys):
        ys = [sum(int(r[t + "_success"]) for r in cases[k]) for t in ARMS]
        ax.plot([i - 0.12, i + 0.12], ys, color="#a2aba8", lw=2)
        for j, (val, color) in enumerate(zip(ys, (GREEN, BLUE), strict=True)):
            ax.scatter(
                i + (-0.12, 0.12)[j],
                val,
                color=color,
                s=48,
                marker=("o", "D")[j],
                label=NAMES[j] if i == 0 else None,
                zorder=3,
            )
    ax.set(
        xticks=np.arange(len(keys)),
        xticklabels=[
            str(i + 1) + ("*" if cases[k][0]["flagged"] else "")
            for i, k in enumerate(keys)
        ],
        yticks=range(7),
        ylabel="Correct judgments of six",
        xlabel="Frozen story, sorted by record identifier",
        ylim=(-0.35, 6.55),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(P / "proof-curriculum-stories.svg", metadata={"Date": None})
    plt.close(fig)
    fig, (tx, bx) = plt.subplots(1, 2, figsize=(7.65, 2.5), layout="constrained")
    for k, name, col in zip(ARMS, NAMES, (GREEN, BLUE), strict=True):
        wall = np.cumsum([r[k + "_wall_s"] for r in rows])
        tx.plot(np.arange(1, len(rows) + 1), wall, label=name, color=col, lw=1.8)
        first = sorted(r[k + "_first_s"] * 1000 for r in rows)
        bx.step(
            first, np.arange(1, len(first) + 1) / len(first), where="post", color=col
        )
    tx.set(xlabel="Request in frozen order", ylabel="Cumulative seconds")
    bx.set(
        xlabel="Time to first token (ms, log scale)",
        ylabel="Cumulative share",
        xscale="log",
    )
    tx.legend(frameon=False, fontsize=8)
    for x in (tx, bx):
        x.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "proof-curriculum-costs.svg", metadata={"Date": None})
    plt.close(fig)

    def pair(field):
        return [arms[k]["groups"]["all"][field] for k in ARMS]

    def textrow(label, ls):
        return [label, *[str(v) for v in ls]]

    cost = [
        textrow("End-to-end seconds", [f"{v:.2f}" for v in pair("wall_s")]),
        textrow("Sum of first-token waits (s)", [f"{v:.3f}" for v in pair("first_s")]),
        textrow("MLX peak (GiB)", [f"{v / 2**30:.3f}" for v in pair("peak")]),
        textrow(
            "Pooled opening tokens/s", [f"{arms[k]['opening_tps']:,.1f}" for k in ARMS]
        ),
        textrow(
            "Subsequent generated tokens/s",
            [f"{arms[k]['decode_tps']:.2f}" for k in ARMS],
        ),
        textrow("Generated tokens", [f"{v:,}" for v in pair("output_tokens")]),
        textrow("Completed requests (of 96)", pair("completed")),
        textrow("Grounded correct (of 96)", pair("grounded")),
    ]
    for arm in ARMS:
        for group in groups:
            rs = [r for r in rows if r["family"] == group]
            assert (
                sum(r[arm + "_success"] for r in rs)
                == arms[arm]["groups"][group]["grounded"]
            )

    def get(arm, metric, group="all"):
        return by[group][arm][metric]

    interval = by["all"]["difference_95_interval"]
    wheel = [
        sum(r[k + "_observed"] for r in rows if r["family"] == "wheel") for k in ARMS
    ]
    file = [
        sum(r[k + "_observed"] for r in rows if r["family"] == "file") for k in ARMS
    ]
    review = [blind["ratings"][k] for k in ARMS]
    logic = [v["reasoning"] for v in (a, b)]

    def worth(v):
        return f"{v:+.1f}"

    chosen = next(w for w in panel["rows"] if w["family"] == "wheel")
    example = next(r for r in rows if r["id"] == chosen["id"])
    previous = read("docs/measurements/file-evidence-long-replay.json")["requests"]
    assert all(r["actual_initial_tokens"] == 131084 for r in previous)
    decisions = {k: v for k, v in d["gates"].items() if not v}
    data = {
        "abstract": f"On 72 held human judgments the loop grounded {get('candidate', 'grounded')} versus {get('control', 'grounded')} for ordinary Base. The paired difference was {d['net']:+d} judgments across 12 stories (two-sided exact p = {d['exact_story_p']:.4g}). It solved every judgment on {a['whole_stories']} versus {b['whole_stories']} whole stories and both practical checks on {a['wheel_pairs']} versus {b['wheel_pairs']} package pairs. All 96 requests took {a['groups']['all']['wall_s']:.2f} versus {b['groups']['all']['wall_s']:.2f} seconds.",
        "decision": (
            "The model met every registered direct-comparison criterion. Matched-depth attribution and independent field qualification are required before promotion."
            if d["passed"]
            else "The direct comparison missed the registered promotion threshold. The gates that missed were "
            + ", ".join(decisions)
            + ". This closes replication, matched-depth training and 128K qualification for these weights."
        ),
        "cost": cost,
        "exposure": f"Native AdamW updated {fit['trainable']:,} parameters over {fit['updates']} steps. The independently reconstructed draws comprise {exposure['sampled_turns']:,} example-turn exposures, {fit['targets']:,} response tokens and {fit['explanation_targets']:,} nonblank explanation-token exposures. The source preparation stage took {fit['prepare_seconds']:.2f} s and optimization {fit['optimizer_seconds']:.2f} s at {fit['peak'] / 2**30:.2f} GiB peak. All {len(hist)} optimizer snapshots were verified.",
        "uncertainty": f"The difference in mean accuracy across whole stories was {worth(100 * (d['net'] / 72))} percentage points. The descriptive 95% paired story-bootstrap interval was [{worth(100 * interval[0])}, {worth(100 * interval[1])}] points. On the seven stories without previously flagged annotation issues, the loop grounded {get('candidate', 'grounded', 'unflagged')}/42 and Base {get('control', 'grounded', 'unflagged')}/42 (paired p = {by['unflagged']['paired_p']:.4g}).",
        "evidence": f"The loop retrieved the correct record on {file[0]}/36 logical file requests versus {file[1]}/36 for Base. It retrieved the declared constraint on {wheel[0]}/16 practical tasks versus {wheel[1]}/16. Each of the eight paired practical cases fixes the interpreter version and changes the distribution so the answer reverses.",
        "reasoning": f"Of completed logical judgments, {logic[0]['nontrivial']}/{logic[0]['completed']} loop replies and {logic[1]['nontrivial']}/{logic[1]['completed']} Base replies contained at least four reasoning words. Across the 48 decisive true/false requests, the loop explicitly numbered premises on {get('candidate', 'numbered_citations')} and Base on {get('control', 'numbered_citations')}. The entire cited set contained a formally sufficient support and the grounded answer was correct on {get('candidate', 'grounded_with_support')} versus {get('control', 'grounded_with_support')}.",
        "blind": f"The blind sample covered 20 preselected items on both models and {blind['distinct']} distinct prompt/evidence/reply combinations. The reader marked {review[0].get('supported', 0)} loop explanations and {review[1].get('supported', 0)} Base explanations fully supported. A reader could sometimes infer model identity from writing style.",
        "ratings": [
            textrow(t.capitalize(), [v.get(t, 0) for v in review])
            for t in ("supported", "partial", "incorrect", "absent", "ambiguous")
        ],
        "example": f"The first enrolled practical request asked whether {chosen['questions'][0].split('Claim: ')[1].split(chr(10))[0]} The published field was {chosen['observation']['value']}. The reference verdict was {chosen['gold']}. The loop’s completed verdict was {example['candidate_verdict'] or 'unavailable'} and Base’s was {example['control_verdict'] or 'unavailable'}.",
        "training": fit["weights"],
        "context": "Release 05 separately retrieved both changed keys with 131,084 actual prompt tokens. The physical long-context result belongs to that qualified release. The present proof-supervised checkpoint has its own acceptance and verification record.",
    }
    save(P / "proof-curriculum-paper.json", data)
    edition(P, "proof-curriculum")


if __name__ == "__main__":
    build()
