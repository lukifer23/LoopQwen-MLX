"""Build the preference-study figures and tables from the enrolled requests and native traces."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition
from scripts.valid_return_analysis import ARMS, ROOT

P = Path("paper")
FAMILIES = ("inline", "file", "wheel", "agenda", "retention")
LABELS = ("Inline human", "File human", "Python", "Schedule", "Retained")
NAMES = ("Trained loop", "Untouched Base")
COLORS = ("#16695f", "#6a619a")


def figure(fig, name):
    fig.savefig(P / f"valid-return-{name}.svg", metadata={"Date": None})
    plt.close(fig)


def historical_evidence(current):
    def measured(study):
        return read(Path("research/results") / study / "decision.json")

    quoref = measured("cross-view-reading-01")["groups"]
    proof = measured("proof-curriculum-01")
    field = measured("field-selection-01")
    panels = [
        (label, 40, *[quoref[k][family]["grounded"] for k in ARMS])
        for label, family in (("Quoref, inline", "inline"), ("Quoref, file", "file"))
    ]
    for label, result in (
        ("FOLIO, both formats", proof),
        ("InfoTabS α3, both formats", field),
        ("InfoTabS α1, both formats", current),
    ):
        panels.append(
            (label, 72, *[sum(result["arms"][k]["by_label"].values()) for k in ARMS])
        )
    fig, ax = plt.subplots(figsize=(7.15, 2.65), layout="constrained")
    for y, (label, denom, a, b) in enumerate(reversed(panels)):
        ax.plot(
            [100 * a / denom, 100 * b / denom], [y, y], color="#a9b2af", linewidth=1.6
        )
        for value, mark, col, n in zip((a, b), ("o", "s"), COLORS, NAMES, strict=True):
            ax.scatter(
                100 * value / denom,
                y,
                marker=mark,
                color=col,
                label=n if y == 0 else None,
                zorder=3,
            )
        ax.text(
            max(100 * a / denom, 100 * b / denom) + 2,
            y,
            f"{a} vs {b} / {denom}",
            va="center",
            fontsize=8,
        )
    ax.set(
        yticks=range(len(panels)),
        yticklabels=[r[0] for r in reversed(panels)],
        xlabel="Grounded answers (%)",
        xlim=(0, 100),
        ylim=(-0.75, len(panels) - 0.35),
    )
    ax.legend(frameon=False, ncols=2, loc="upper right")
    figure(fig, "trajectory")
    depth = []
    for title, n, denom in (
        ("Induction, original panel", "native-induction-anchored-matched-01", 64),
        ("Induction, fresh panel", "native-induction-anchored-replication-01", 64),
        ("Typed public tool calls", "tool-dialogue-typed-depth-01", 22),
        ("Four tool panels", "tool-dialogue-counterfactual-depth-01", 132),
    ):
        d = measured(n)
        if "scores" in d:
            vals = [d["scores"][k]["correct"] for k in ("recurrent", "dense")]
            p = d["p"]
        else:
            vals = [d["metrics"][k]["success"] for k in ARMS]
            p = d["p_exact"]
        depth.append([title, *[f"{v}/{denom}" for v in vals], f"{p:.5g}"])
    return depth


def build():
    spec = read(ROOT / "registration.json")
    fit = read(ROOT / "fit.json")
    odds = unpack(ROOT / "reference-logodds.json.gz")
    views = unpack(ROOT / "training-views.json.gz")
    data = read("docs/measurements/valid-return-analysis.json")
    sources = read("docs/measurements/valid-return-provenance.json")
    exp = read("docs/measurements/valid-return-exposure.json")
    decision = read(ROOT / "decision.json")
    panel = read(spec["material"])
    with (P / "valid-return-requests.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 112 and len(odds) == 1272
    sides = [decision["arms"][a] for a in ARMS]
    groups = [a["groups"]["all"] for a in sides]
    analyses = [data["arms"][a] for a in ARMS]
    plt.rcParams.update(
        {
            "svg.hashsalt": "valid-return-01",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.05, 2.7), layout="constrained")
    centers = np.arange(5)
    family_rows = []
    for i, (sid, name, color) in enumerate(zip(sides, NAMES, COLORS, strict=True)):
        g = [sid["groups"][f] for f in FAMILIES]
        bars = ax.bar(
            centers + (i - 0.5) * 0.38,
            [w["grounded"] / w["requests"] for w in g],
            0.36,
            color=color,
            label=name,
        )
        ax.bar_label(
            bars,
            labels=[f"{w['grounded']}/{w['requests']}" for w in g],
            fontsize=8,
            padding=3,
        )
    ax.set(
        xticks=centers,
        xticklabels=LABELS,
        ylim=(0, 1.15),
        ylabel="Grounded success rate",
    )
    ax.legend(frameon=False, ncols=2, loc="upper center")
    figure(fig, "outcomes")
    for f, label in zip(FAMILIES, LABELS, strict=True):
        family_rows.append(
            [label]
            + [
                f"{s['groups'][f]['grounded']} / {s['groups'][f]['completed']} / {s['groups'][f]['requests']}"
                for s in sides
            ]
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), layout="constrained")
    milestone_rows = []
    for ax, name, family, denom in zip(
        axes,
        ("Package compatibility", "Calendar-dependent weekday"),
        ("wheel", "weekday"),
        (16, 8),
        strict=True,
    ):
        subset = (
            [r for r in rows if r["family"] == "wheel"]
            if family == "wheel"
            else [r for r in rows if r["id"].endswith("-weekday")]
        )
        measures = (
            "first_valid_wrong",
            "first_milestone",
            "grounded_sequence",
            "success",
        )
        assert len(subset) == denom
        for i, (a, col) in enumerate(zip(ARMS, COLORS, strict=True)):
            vals = [sum(r[a + "_" + k] == "True" for r in subset) for k in measures]
            bars = ax.bar(
                np.arange(4) + (i - 0.5) * 0.39, vals, 0.37, color=col, label=NAMES[i]
            )
            ax.bar_label(bars, padding=2, fontsize=8)
            milestone_rows.append([name, NAMES[i], *map(str, vals)])
        ax.set(
            title=name,
            ylim=(0, denom + 2),
            xticks=range(4),
            xticklabels=(
                "Valid wrong\nopening",
                "Needed\nlookup",
                "Complete\nsequence",
                "Grounded\nanswer",
            ),
            ylabel="Requests",
        )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure(fig, "milestones")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    refstats = []
    for i, kind in enumerate(("field", "operation", "verdict")):
        arr = np.array(
            [
                odds[w["pair"]]
                for w in views
                if w["kind"] == kind and w["choice"] == "preferred"
            ]
        )
        refstats.append(
            [kind, str(len(arr)), f"{np.median(arr):.2f}", f"{sum(arr > 0)}/{len(arr)}"]
        )
        axes[0].plot(np.sort(arr), np.arange(1, len(arr) + 1) / len(arr), label=kind)
    axes[0].axvline(0, color="#666", lw=0.7, ls=":")
    axes[0].set(
        xlabel="Reference log odds (nats)",
        ylabel="Fraction of preferences",
        title="Before optimization",
    )
    axes[0].legend(frameon=False)
    loss = fit["loss"]
    axes[1].plot(
        [b["step"] for b in loss],
        [b["loss"] for b in loss],
        marker="o",
        color=COLORS[0],
    )
    axes[1].set(
        xlabel="AdamW update",
        ylabel="Mixed batch loss",
        title="Sampled training batches",
    )
    figure(fig, "learning")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ("first_s", "wall_s"),
        ("Time to first token (s)", "End-to-end request (s)"),
        strict=True,
    ):
        for a, n, c in zip(ARMS, NAMES, COLORS, strict=True):
            x = sorted(float(r[a + "_" + metric]) for r in rows)
            ax.step(
                x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, label=n
            )
        ax.set(xscale="log", xlabel=label, ylabel="Fraction of requests")
    axes[0].legend(frameon=False)
    figure(fig, "cost")

    def pair(key):
        return [str(v[key]) for v in sides]

    def fmt(key, precision=2, scale=1):
        return [f"{v[key] / scale:.{precision}f}" for v in groups]

    costs = [
        ["Completed requests", *fmt("completed", 0)],
        ["Generated tokens", *fmt("output_tokens", 0)],
        ["Total wall time (s)", *fmt("wall_s")],
        ["Sum of first-token waits (s)", *fmt("first_s")],
        ["Peak MLX allocation (GiB)", *fmt("peak", 3, 2**30)],
        ["Opening throughput (tokens/s)", *[f"{v['opening_tps']:.1f}" for v in sides]],
        ["Decode throughput (tokens/s)", *[f"{v['decode_tps']:.1f}" for v in sides]],
    ]
    human = [sum(v["by_label"].values()) for v in sides]
    ci = [100 * v for v in data["human_cluster_interval"]]
    bylabel = [
        [k.capitalize(), *[str(v["by_label"].get(k, 0)) for v in sides]]
        for k in ("true", "false", "uncertain")
    ]
    reviews = [
        [k.capitalize(), *[str(a["rating"].get(k, 0)) for a in analyses]]
        for k in ("supported", "partial", "incorrect", "absent", "ambiguous")
    ]
    reg = [k.replace("_", " ") for k, v in decision["gates"].items() if not v]
    wheel = next(r for r in rows if r["family"] == "wheel")
    agenda = next(r for r in rows if r["id"].endswith("-weekday"))

    def actual(row):
        refs = json.loads(row["required"])
        stream = json.loads(row["candidate_calls"])

        def call(e):
            a = e["call"]["arguments"]
            if "value_column" in a:
                return f"{a.get('key_column')} → {a.get('value_column')}"
            return e["call"]["name"] + "(" + str(a.get("date", "")) + ")"

        return {
            "question": row["question"].split("\n\n")[0],
            "claim": row["question"].split("\n\n")[-1],
            "expected": ", ".join(
                call(e) + " = " + str(e["observation"].get("value", e["observation"]))
                for e in refs
            ),
            "actual": ", ".join(
                call(e)
                + " = "
                + str(e["observation"].get("value", e["observation"]))[:90]
                for e in stream
            )
            or "No executed call",
            "answer": row["candidate_answer"][:140] or "No completed answer",
            "gold": row["gold"],
        }

    def preferred_at_start(kind, source):
        selected = [
            odds[w["pair"]]
            for w in views
            if w["kind"] == kind
            and w["source"] == source
            and w["choice"] == "preferred"
        ]
        return f"{sum(v > 0 for v in selected)}/{len(selected)}"

    initial_choices = (
        f"With both full responses provided, the starting model preferred the correct calendar operation on {preferred_at_start('operation', 'agenda')} pairs. "
        f"When the source-specific rationale was supplied, its package and schedule verdict margins favored the target on {preferred_at_start('verdict', 'package')} and {preferred_at_start('verdict', 'agenda')} pairs. "
        f"The corresponding human-table verdict count was {preferred_at_start('verdict', 'human')}. These are conditional training measurements. The direct evaluation requires the model to generate its own reasoning and select every action."
    )
    depth = historical_evidence(decision)
    ballots = read("docs/measurements/valid-return-ballot-sensitivity.json")
    bt = [ballots[arm]["totals"] for arm in ARMS]
    ballot_text = (
        f"Sixteen of the 36 file claims had at least one dissenting field-selection ballot. "
        f"Requiring the majority field gave {bt[0]['registered']} versus {bt[1]['registered']} grounded answers. "
        f"Accepting any fully satisfied individual in-table ballot gave {bt[0]['votes_1']} versus {bt[1]['votes_1']}. "
        f"Requiring three satisfied ballots gave {bt[0]['votes_3']} versus {bt[1]['votes_3']}. "
        "All of a voter's selected fields must be retrieved, and the final verdict must still be correct."
    )
    save(
        P / "valid-return-paper.json",
        {
            "historical_depth": depth,
            "initial_choices": initial_choices,
            "deadlines": [
                [
                    f"{seconds:g} s",
                    *[
                        str(
                            sum(
                                r[arm + "_success"] == "True"
                                and float(r[arm + "_wall_s"]) <= seconds
                                for r in rows
                            )
                        )
                        for arm in ARMS
                    ],
                ]
                for seconds in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
            ],
            "ballot_sensitivity": ballot_text,
            "abstract": f"The loop grounded {human[0]}/72 human judgments versus {human[1]}/72 for Base. Across all 112 requests it grounded {groups[0]['grounded']} versus {groups[1]['grounded']} in {groups[0]['wall_s']:.2f} versus {groups[1]['wall_s']:.2f} seconds. The registered joint decision was {'pass' if decision['passed'] else 'fail'}.",
            "humans": f"The paired difference was {decision['net']:+d} of 72 judgments. A two-sided exact sign-flip test over 12 whole tables gave p = {decision['exact_story_p']:.5f}. The 95% table bootstrap interval was {ci[0]:+.1f} to {ci[1]:+.1f} percentage points.",
            "families": family_rows,
            "milestones": milestone_rows,
            "bylabel": bylabel,
            "reviews": reviews,
            "costs": costs,
            "references": refstats,
            "complete": [
                ["Human tables (of 12)", *pair("whole_stories")],
                ["Package pairs (of 8)", *pair("wheel_pairs")],
                ["Agenda pairs (of 8)", *pair("agenda_pairs")],
            ],
            "gates": ", ".join(reg) or "All registered gates passed.",
            "fit": f"The {fit['updates']} updates used {exp['preferred_draws']:,} preferred trajectories, {exp['preferred_tokens']:,} preferred tokens and {exp['rejected_tokens']:,} rejected tokens. Of the preferred tokens {exp['explanation_tokens']:,} were nonblank tokens preceding the reasoning close on verdict examples. Prefix and reference preparation took {fit['prepare_seconds']:.1f} s, optimization {fit['optimizer_seconds']:.1f} s, and the training peak was {fit['peak'] / 2**30:.2f} GiB.",
            "source": f"All {sources['selected_claims']} sampled human source claims and their ballots are preserved. Masking titles and numbers produced {sources['masked_exact_claims']} exact held/training text matches. The maximum five-word shingle Jaccard similarity was {sources['maximum_masked_shingle_jaccard']:.2f}.",
            "sensitivity": [
                [k, *[str(sum(a[f][k] for f in ("file", "inline"))) for a in analyses]]
                for k in ("success", "bare", "final_sentence")
            ],
            "trade": f"The loop newly grounded {len(data['gained'])} requests and lost {len(data['lost'])} Base successes. On {len(data['jointly_faster_gains'])} newly correct requests it also reduced first-token wait, end-to-end time and peak allocation together.",
            "costwins": [
                [
                    label,
                    str(
                        sum(
                            float(r["candidate_" + m]) < float(r["control_" + m])
                            for r in rows
                        )
                    ),
                ]
                for m, label in (
                    ("first_s", "First token"),
                    ("wall_s", "End to end"),
                    ("peak", "Peak allocation"),
                )
            ],
            "wheel": actual(wheel),
            "agenda": actual(agenda),
            "projects": ", ".join(
                v["package"] + " " + v["version"] for v in panel["wheels"]
            ),
            "grounding": [
                [
                    f,
                    *[
                        str(a[f]["first_valid_wrong"]) + " / " + str(a[f]["recovered"])
                        for a in analyses
                    ],
                ]
                for f in ("file", "wheel", "agenda")
            ],
            "dates": dict(
                Counter(
                    w["date"]
                    for w in panel["agenda"]
                    if w["id"]
                    in {row["cluster"] for row in rows if row["family"] == "agenda"}
                )
            ),
        },
    )
    edition(P, "valid-return")
    edition(P, "valid-return", verify=True)


if __name__ == "__main__":
    build()
