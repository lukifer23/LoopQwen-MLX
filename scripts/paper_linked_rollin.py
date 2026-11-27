"""Typeset the held paired trajectories, training evidence, and installed runtime measurements."""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition
from scripts.linked_rollin_analysis import ARMS, DOC, PLAN, ROOT
from scripts.paper_valid_return import COLORS, NAMES
from scripts.linked_rollin_retention import compare as retention

P = Path("paper")
GROUPS = {
    "linked_inline": "Inline table",
    "linked_file": "Table file",
    "inline": "Prior inline",
    "file": "Prior file",
    "wheel": "Software",
    "agenda": "Schedule",
    "retention": "Lookup",
}


def figout(fig, name):
    fig.savefig(P / f"linked-rollin-{name}.svg", metadata={"Date": None})
    plt.close(fig)


def build():
    d = read(ROOT / "decision.json")
    f = read(ROOT / "fit.json")
    analysis = read(DOC / "linked-rollin-analysis.json")
    exposure = read(DOC / "linked-rollin-exposure.json")
    plan = read(PLAN)
    review = {w["cluster"]: w for w in plan["items"]}
    with (P / "linked-rollin-requests.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    assert len(cases) == 84 and len(review) == 24
    results = [d["arms"][a] for a in ARMS]
    current = analysis["sources"]["all_success"]
    plt.rcParams.update(
        {
            "svg.hashsalt": "linked-rollin-02",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.25, 2.8), layout="constrained")
    for i, (sid, n, color) in enumerate(zip(results, NAMES, COLORS, strict=True)):
        groups = [sid["families"][k] for k in GROUPS]
        bars = ax.bar(
            np.arange(len(groups)) + (i - 0.5) * 0.38,
            [g["grounded"] / g["requests"] for g in groups],
            0.36,
            label=n,
            color=color,
        )
        ax.bar_label(
            bars, labels=[str(g["grounded"]) for g in groups], fontsize=8, padding=2
        )
    ax.set(
        xticks=range(len(GROUPS)),
        xticklabels=list(GROUPS.values()),
        ylim=(0, 1.09),
        ylabel="Grounded success per request",
    )
    ax.legend(frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.19))
    figout(fig, "outcomes")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 2.85), layout="constrained", sharey=True
    )
    keys = ["first_required", "waited_for_source", "official_exact", "success"]
    for ax, (group, title) in zip(axes, list(GROUPS.items())[:2], strict=True):
        for j, (arm, n, color) in enumerate(zip(ARMS, NAMES, COLORS, strict=True)):
            vals = analysis["families"][group][arm]
            ax.bar(
                np.arange(4) + (j - 0.5) * 0.38,
                [vals[k] for k in keys],
                0.36,
                color=color,
                label=n,
            )
        ax.set(
            xticks=range(4),
            xticklabels=["First call", "Source in\norder", "Exact\nanswer", "Both"],
            ylim=(0, 25),
            title=title,
        )
    axes[0].set_ylabel("Requests out of 24")
    axes[0].legend(frameon=False, fontsize=8)
    figout(fig, "evidence")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), layout="constrained")
    statuses = [
        "answered",
        "turn_limit",
        "token_limit",
        "invalid_call",
        "unfinished_thinking",
    ]
    enroll = []
    for phase in range(2):
        records = unpack(ROOT / f"student-round-{phase}.json.gz")
        requests = [q for w in records for q in w["requests"]]
        counts = Counter(q["trajectory"]["status"] for q in requests)
        enroll.append(
            [
                str(phase * 400),
                str(len(requests)),
                str(counts["answered"]),
                str(sum(q["score"]["success"] for q in requests)),
                str(f["phases"][phase]["corrections"]),
                str(f["phases"][phase]["with_real_alternative_token"]),
            ]
        )
        axes[0].bar(
            np.arange(len(statuses)) + (phase - 0.5) * 0.38,
            [counts[k] for k in statuses],
            0.36,
            color=COLORS[phase],
            label=f"Step {phase * 400}",
        )
    axes[0].set(
        xticks=range(5),
        xticklabels=["Answer", "Turns", "Tokens", "Invalid", "Reason"],
        ylabel="Training requests",
        title="Free trainee responses",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        [r["step"] for r in f["loss"]],
        [r["loss"] for r in f["loss"]],
        color=COLORS[0],
        marker="o",
        markersize=3,
    )
    axes[1].set(
        xlabel="Native optimizer step",
        ylabel="Sampled loss",
        title="Mixed training objective",
    )
    figout(fig, "learning")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    for ax, k, label in zip(
        axes,
        ("first_s", "wall_s"),
        ("First generated token (s)", "End-to-end latency (s)"),
        strict=True,
    ):
        for arm, n, color in zip(ARMS, NAMES, COLORS, strict=True):
            vals = sorted(float(row[arm + "_" + k]) for row in cases)
            ax.step(
                vals,
                np.arange(1, len(vals) + 1) / len(vals),
                where="post",
                color=color,
                label=n,
            )
        ax.set(xscale="log", xlabel=label, ylabel="Fraction of requests")
    axes[1].legend(frameon=False, fontsize=8)
    figout(fig, "cost")
    deadlines = [0.5, 1, 2, 5, 10, 30]
    deadline_table = []
    for limit in deadlines:
        deadline_table.append(
            [
                f"{limit:g} s",
                *[
                    str(
                        sum(
                            row[a + "_success"] == "True"
                            and float(row[a + "_wall_s"]) <= limit
                            for row in cases
                        )
                    )
                    for a in ARMS
                ],
            ]
        )
    common = [row for row in cases if all(row[a + "_success"] == "True" for a in ARMS)]
    delivery = []
    for a, side in zip(ARMS, results, strict=True):
        delivery.append(f"{60 * side['all']['grounded'] / side['all']['wall_s']:.2f}")
    common_text = (
        (
            f"On the {len(common)} requests both models solved, summed wait was "
            + " versus ".join(
                f"{sum(float(row[a + '_wall_s']) for row in common):.2f} s"
                for a in ARMS
            )
            + "."
        )
        if common
        else "The two models had no solved requests in common."
    )
    prices = [
        [lab, *[f"{z['all'][k] / scale:,.{prec}f}" for z in results]]
        for lab, k, scale, prec in (
            ("Grounded answers", "grounded", 1, 0),
            ("Completed responses", "completed", 1, 0),
            ("Output tokens", "output_tokens", 1, 0),
            ("Summed first-token latency (s)", "first_s", 1, 3),
            ("Total user wait (s)", "wall_s", 1, 1),
            ("Peak MLX allocation (GiB)", "peak", 2**30, 3),
        )
    ]
    prices.extend(
        [
            [lab, *[f"{z[k]:,.2f}" for z in results]]
            for lab, k in (
                ("Prompt rate (tokens/s)", "opening_tps"),
                ("Decode rate (tokens/s)", "decode_tps"),
            )
        ]
    )
    physical = []
    for arm in ARMS:
        observations = unpack(ROOT / (arm + ".json.gz"))
        turns = [
            t
            for r in observations
            for q in r["requests"]
            for t in q["trajectory"]["turns"]
        ]
        logical = sum(len(t["prompt_ids"]) for t in turns)
        reused = sum(t.get("reused_prompt_tokens", 0) for t in turns)
        physical.append(
            (logical, reused, (logical - reused) / sum(t["ttft_s"] for t in turns))
        )
    prices.extend(
        [
            [label, *[f"{vals[i]:,.{precision}f}" for vals in physical]]
            for i, (label, precision) in enumerate(
                (
                    ("All rendered prompt tokens", 0),
                    ("Previously cached prompt tokens", 0),
                    ("Uncached prompt rate (tokens/s)", 2),
                )
            )
        ]
    )
    scores = []
    for setname, prefix in (
        ("All 24 questions", "all"),
        ("20 directly supported", "direct_support"),
    ):
        for label, key in (
            ("Primary", "success"),
            ("Variant aliases", "variant_grounded"),
            ("Ordered reads", "waited_for_source"),
        ):
            v = analysis["sources"][prefix + "_" + key]
            scores.append(
                [
                    setname,
                    label,
                    *[f"{v[a]}/{v['requests']}" for a in ARMS],
                    f"{v['paired_p']:.4g}",
                ]
            )
    reads = []
    for key in (
        "equivalent",
        "appropriately qualified",
        "incomplete",
        "contradicted",
        "absent",
        "invented_claims",
        "source_observed_and_supported",
    ):
        reads.append(
            [
                key.replace("_", " ").capitalize(),
                *[
                    str(
                        analysis["blinded"][a].get(
                            key, analysis["blinded"][a]["counts"].get(key, 0)
                        )
                    )
                    for a in ARMS
                ],
            ]
        )
    whole = [
        [GROUPS.get(k, k.capitalize()), *[str(z["complete"][k]) for z in results]]
        for k in ("linked", "wheel", "agenda")
    ]
    # Fixed illustrations: first source in the pre-generation review for each presentation.
    exemplars = []
    for mode in ("inline", "file"):
        source = next(
            w for w in plan["items"] if w["blind_sample"].endswith("-" + mode)
        )
        row = next(r for r in cases if r["id"] == source["blind_sample"])
        detail = []
        for arm in ARMS:
            ans = row[arm + "_answer"].strip()
            detail.append(
                f"{row[arm + '_status']} · {row[arm + '_callbacks']} calls · "
                + (
                    ans[:220] + ("…" if len(ans) > 220 else "")
                    if ans
                    else "no final text"
                )
            )
        exemplars.append([source["question"], source["published"], *detail])
    r = read("research/results/cache-state-conversations-01/decision.json")["totals"]
    retained = d["earlier_base_successes_lost"]
    ci = analysis["timing_intervals"]
    history = []
    for title, study, n in (
        ("Three-fact induction, original", "native-induction-anchored-matched-01", 64),
        (
            "Three-fact induction, replication",
            "native-induction-anchored-replication-01",
            64,
        ),
        ("Typed public function calls", "tool-dialogue-typed-depth-01", 22),
        ("Four tool panels", "tool-dialogue-counterfactual-depth-01", 132),
    ):
        fixed = read(Path("research/results") / study / "decision.json")
        if "scores" in fixed:
            v = [fixed["scores"][k] for k in ("recurrent", "dense")]
            key = "correct"
        else:
            v = [fixed["metrics"][k] for k in ARMS]
            key = "success"
        history.append(
            [
                title,
                *[f"{w[key]}/{n}" for w in v],
                " / ".join(f"{w['seconds']:.2f}" for w in v),
            ]
        )
    systems = []
    for title, study in (
        ("FOLIO", "proof-curriculum-01"),
        ("InfoTabS cross-domain", "field-selection-01"),
        ("InfoTabS ordinary", "valid-return-01"),
    ):
        previous = read(Path("research/results") / study / "decision.json")
        v = [previous["arms"][k] for k in ARMS]
        systems.append(
            [
                title,
                *[f"{sum(w['by_label'].values())}/72" for w in v],
                f"{previous['exact_story_p']:.4g}",
                " / ".join(f"{w['groups']['all']['wall_s']:.1f}" for w in v),
            ]
        )
    systems.append(
        [
            "HybridQA, present study",
            *[f"{current[a]}/48" for a in ARMS],
            f"{current['paired_p']:.4g}",
            " / ".join(f"{w['all']['wall_s']:.1f}" for w in results),
        ]
    )
    save(
        P / "linked-rollin-paper.json",
        dict(
            retention=retention(),
            deadlines=deadline_table,
            delivery=" and ".join(delivery),
            common=common_text,
            history=history,
            systems=systems,
            abstract=f"The trained loop gave {current['candidate']} grounded exact answers on 48 open requests. Base gave {current['control']}. The net change was {current['net']:+d} requests (whole-source paired p = {current['paired_p']:.4g}). "
            f"Across all 84 requests the trained model completed {results[0]['all']['completed']} and Base completed {results[1]['all']['completed']}. "
            + (
                "The full registered comparison passed."
                if d["passed"]
                else "The full registered comparison missed its promotion gate."
            ),
            unmatched=", ".join(
                k.replace("_", " ") for k, v in d["gates"].items() if not v
            )
            or "None",
            phases=enroll,
            fit=f"{f['trainable']:,} trainable parameters; {f['targets']:,} positive token targets, {f['explanation_targets']:,} nonblank explanation tokens and {f['contrast_draws']:,} draws with an observed competing token. Prefix preparation took {f['prepare_seconds']:.1f} s, free training conversations {f['rollin_seconds']:.1f} s and optimization {f['optimizer_seconds']:.1f} s.",
            diversity=f"The sampler visited {exposure['distinct_positive_views_drawn']} distinct training views over 3,200 draws.",
            scores=scores,
            whole=whole,
            costs=prices,
            reviews=reads,
            examples=exemplars,
            f1=" and ".join(f"{z['mean_open_f1']:.3f}" for z in results),
            lost=str(len(retained)),
            lifecycle=f"{len(analysis['gains'])} individual requests gained grounded success and {len(analysis['losses'])} lost it. On {len(analysis['joint_gains'])} of the gains, the loop also reduced first-token wait, total wait and peak allocation on that same request.",
            intervals="; ".join(
                f"{label} {ci[k][0]:.3f}–{ci[k][1]:.3f}"
                for k, label in (
                    ("first_ratio", "first-token ratio"),
                    ("wall_ratio", "wall-time ratio"),
                )
            ),
            practical=[
                [
                    g,
                    *[str(analysis["families"][g][a]["success"]) for a in ARMS],
                    str(results[0]["families"][g]["requests"]),
                ]
                for g in ("wheel", "agenda", "file", "inline", "retention")
            ],
            runtime=f"Cold first-token time totaled {r['stock']['cold_first_s']:.3f} s with the original implementation and {r['state']['cold_first_s']:.3f} s with the cache-state implementation. Complete process time totaled {r['stock']['cold_s']:.3f} and {r['state']['cold_s']:.3f} s. Peak allocation fell {(r['stock']['peak'] - r['state']['peak']) / 2**20:.1f} MiB.",
            sha=f["weights"],
            panel=read(ROOT / "source-qualification.json")["panel_sha256"],
        ),
    )
    edition(P, "linked-rollin")
    edition(P, "linked-rollin", verify=True)


if __name__ == "__main__":
    build()
