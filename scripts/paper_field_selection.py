"""Typeset paired performance, argument binding and cost from all 192 executed requests."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition
from scripts.field_selection_analysis import ARMS, STUDY, extract

P = Path("paper")
NAMES = ("Trained loop", "Untouched Base")
COLORS = ("#146e64", "#576498")
FAMILIES = ("inline", "file", "wheel", "retention")


def save_plot(fig, name):
    fig.savefig(P / ("field-selection-" + name + ".svg"), metadata={"Date": None})
    plt.close(fig)


def build():
    rows, _ = extract(STUDY)
    dec = read(STUDY / "decision.json")
    a = read("docs/measurements/field-selection-analysis.json")
    exposure = read("docs/measurements/field-selection-training-exposure.json")
    fit = read(STUDY / "fit.json")
    panel = read(read(STUDY / "registration.json")["material"])
    oracle = read("docs/measurements/field-selection-oracle.json")
    hist = read(STUDY / "optimizer-steps.json")
    sides = [dec["arms"][k] for k in ARMS]
    groups = [v["groups"]["all"] for v in sides]
    plt.rcParams.update(
        {
            "svg.hashsalt": "field-selection-01",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.4, 2.65), layout="constrained")
    for idx, (name, col, side) in enumerate(zip(NAMES, COLORS, sides, strict=True)):
        dx = x + (idx - 0.5) * 0.38
        vals = [side["groups"][f] for f in FAMILIES]
        bars = ax.bar(
            dx,
            [v["grounded"] / v["requests"] for v in vals],
            0.36,
            color=col,
            label=name,
        )
        ax.bar_label(
            bars, labels=[f"{v['grounded']}/{v['requests']}" for v in vals], padding=3
        )
    ax.set(
        xticks=x,
        xticklabels=[
            "Inline judgments",
            "Tool-grounded judgments",
            "Python compatibility",
            "Retained lookups",
        ],
        ylabel="Grounded success",
        ylim=(0, 1.22),
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_plot(fig, "accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.75), layout="constrained")
    kinds = ("exact", "swapped", "other return field", "other", "none")
    color = ("#187764", "#b45148", "#d3a34c", "#718398", "#d6dce1")
    for ax, family in zip(axes, ("file", "wheel"), strict=True):
        offset = np.zeros(2)
        for kind, c in zip(kinds, color, strict=True):
            sizes = np.array(
                [
                    a["arms"][arm]["families"][family]["first_calls"].get(kind, 0)
                    for arm in ARMS
                ]
            )
            bars = ax.barh([1, 0], sizes, left=offset, color=c, label=kind)
            for b, n in zip(bars, sizes, strict=True):
                if n >= 2:
                    ax.text(
                        b.get_x() + n / 2,
                        b.get_y() + b.get_height() / 2,
                        str(n),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            offset += sizes
        ax.set(
            yticks=[1, 0],
            yticklabels=NAMES,
            title=(
                "Human evidence fields"
                if family == "file"
                else "Published requirements"
            ),
            xlabel="Requests",
            xlim=(0, max(offset)),
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="outside lower center", frameon=False)
    save_plot(fig, "arguments")

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.15), layout="constrained")
    classes = ("true", "false", "uncertain")
    for ax, arm, name in zip(axes, ARMS, NAMES, strict=True):
        conf = a["arms"][arm]["confusion"]
        data = [[conf[k].get(p, 0) for p in (*classes, "none")] for k in classes]
        ax.imshow(data, cmap="Blues", vmin=0, vmax=24)
        ax.set(
            xticks=range(4),
            xticklabels=["True", "False", "Uncertain", "No verdict"],
            yticks=range(3),
            yticklabels=["True", "False", "Uncertain"],
            title=name,
            xlabel="Emitted verdict",
            ylabel="Human verdict",
        )
        ax.tick_params(axis="x", labelsize=8)
        for (i, j), v in np.ndenumerate(np.array(data)):
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > 13 else "#152335",
            )
    save_plot(fig, "confusion")

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.9), layout="constrained")
    for i, (arm, name, col) in enumerate(zip(ARMS, NAMES, COLORS, strict=True)):
        for ax, metric, label in zip(
            axes,
            ("seconds", "first_seconds"),
            ("Complete request (s)", "First token (s)"),
            strict=True,
        ):
            arr = sorted(r[arm + "_" + metric] for r in rows)
            ax.step(
                arr,
                np.arange(1, len(arr) + 1) / len(arr),
                where="post",
                color=col,
                label=name,
            )
            ax.set(
                xlabel=label,
                ylabel="Fraction of requests",
                xscale="log",
                ylim=(0, 1.04),
            )
    axes[0].legend(frameon=False)
    save_plot(fig, "timing")

    interval = a["interval"]

    def pair(values, digits=0):
        return [f"{v:,.{digits}f}" for v in values]

    summary = [
        [label, *values]
        for label, values in [
            ("Grounded requests / 96", pair(v["grounded"] for v in groups)),
            ("Completed requests / 96", pair(v["completed"] for v in groups)),
            ("Whole tables / 12", pair(x["whole_stories"] for x in sides)),
            ("Whole package pairs / 8", pair(x["wheel_pairs"] for x in sides)),
            ("Total measured seconds", pair((v["wall_s"] for v in groups), 2)),
            ("First-token seconds, summed", pair((v["first_s"] for v in groups), 3)),
            ("Peak MLX GiB", pair((v["peak"] / 2**30 for v in groups), 3)),
            (
                "Opening input tokens / second",
                pair((v["opening_tps"] for v in sides), 1),
            ),
            ("Subsequent tokens / second", pair((v["decode_tps"] for v in sides), 2)),
            ("Generated tokens", pair(v["output_tokens"] for v in groups)),
        ]
    ]
    breakdown = []
    for f in ("file", "wheel"):
        for arm, name in zip(ARMS, NAMES, strict=True):
            v = a["arms"][arm]["families"][f]
            breakdown.append(
                [
                    f,
                    name,
                    str(v["observed"]),
                    str(v["correct_after_grounding"]),
                    str(v["repair"]),
                ]
            )
    counts = [sum(x["by_label"].values()) for x in sides]
    verdict = (
        "met every registered direct-comparison threshold"
        if dec["passed"]
        else "missed its joint promotion rule"
    )
    misses = ", ".join(k.replace("_", " ") for k, v in dec["gates"].items() if not v)
    reader = [
        [label, *pair(a["reading"][k].get(label, 0) for k in ARMS)]
        for label in ("supported", "partial", "incorrect", "absent", "ambiguous")
    ]
    m = {
        "abstract": f"The loop grounded {counts[0]}/72 human table judgments and Base grounded {counts[1]}/72. The difference was {100 * dec['net'] / 72:+.1f} percentage points (exact paired p = {dec['exact_story_p']:.4f}, whole-table bootstrap 95% interval {100 * interval[0]:+.1f} to {100 * interval[1]:+.1f}). On 16 checks against published package constraints the systems grounded {sides[0]['groups']['wheel']['grounded']} and {sides[1]['groups']['wheel']['grounded']}. Across all 96 requests they took {groups[0]['wall_s']:.2f} and {groups[1]['wall_s']:.2f} seconds. The trained system {verdict}.",
        "scope": f"The whole-table comparison comprises 12 independent entities. A constant-label responder with perfect evidence access can obtain {a['constant_gold_ceiling']}/72 here. On the {a['joint_completions']} table judgments where both models completed, the loop grounded {a['arms']['candidate']['joint_correct']} and Base grounded {a['arms']['control']['joint_correct']}. Across all four families, {len(a['lost_base_successes'])} individually successful Base outcomes were lost.",
        "fit": f"MLX applied {fit['updates']} AdamW updates to {fit['trainable']:,} shared parameters. Independent reconstruction found {sum(exposure['response_draws'].values()):,} response draws and {exposure['targets']:,} target tokens, including {exposure['human_field_tokens']:,} nonblank tokens identifying human-selected field values on the table triplets. There were {exposure['triplet_steps_by_view']['file']} tool-return triplet updates and {exposure['triplet_steps_by_view']['inline']} inline triplet updates. Prefix preparation took {fit['prepare_seconds']:.1f} seconds and optimization {fit['optimizer_seconds']:.1f} seconds. Training peak was {fit['peak'] / 2**30:.3f} GiB.",
        "reading": reader,
        "summary": summary,
        "lookups": breakdown,
        "decision": (
            "The pair passed every registered threshold."
            if dec["passed"]
            else "The following registered gates failed: " + misses + "."
        ),
        "integrity": f"An independent SQLite interpreter checked {sum(oracle['sqlite_held'].values())} held lookup cells and {oracle['opposite_training_roles']} opposite-direction training queries. Every intentionally swapped search missed in the original table. All eight original wheels were downloaded in full ({sum(w['artifact_bytes'] for w in oracle['held_real_wheels']):,} bytes). Each embedded METADATA member matched its separately published sidecar byte for byte. The verifier examined {len(hist)} optimizer snapshots and compared the final trainable tensors and AdamW state.",
        "packages": ", ".join(
            w["package"] + " " + w["version"] for w in panel["wheels"]
        ),
    }
    save(P / "field-selection-paper.json", m)
    edition(P, "field-selection")
    edition(P, "field-selection", verify=True)


if __name__ == "__main__":
    build()
