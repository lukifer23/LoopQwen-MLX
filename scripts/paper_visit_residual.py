"""Publish source-paired reasoning, grounded software decisions and measured second-visit effects."""

import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from scripts.assemble_evidence_paper import edition
from scripts.paper_extended_prefix import build as prefix
from scripts.visit_residual_reading import DOC, PANEL, ROOT
from scripts.visit_residual_reading import build as reading

P = Path("paper")
ARMS = ("candidate", "control")
FAMILIES = {
    "effects": "Human causal questions",
    "dependency": "New project releases",
    "same_version": "Fixed-version swaps",
    "wheel": "Earlier manifests",
    "agenda": "Earlier schedules",
    "file": "Earlier file judgments",
    "inline": "Earlier inline judgments",
    "retention": "Earlier lookups",
}
COLORS = ("#1c7061", "#715986")


def csv_file(stem, rows):
    with (P / (stem + ".csv")).open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def build():
    reading("report")
    prefix()
    d = read(ROOT / "decision.json")
    fit = read(ROOT / "fit.json")
    a, b = [d["arms"][k] for k in ARMS]
    panel = read(PANEL)["rows"]
    runs = [unpack(ROOT / (arm + ".json.gz")) for arm in ARMS]
    requests = []
    for w, x, y in zip(panel, *runs, strict=True):
        assert w["id"] == x["id"] == y["id"]
        rec = {"id": w["id"], "source": w["cluster"], "family": w["family"]}
        for arm, case in zip(ARMS, (x, y), strict=True):
            q = case["requests"][0]
            tr = q["trajectory"]
            rec.update(
                {
                    arm + "_" + k: v
                    for k, v in {
                        "success": int(q["score"]["success"]),
                        "status": tr["status"],
                        "f1": q["score"].get("f1", ""),
                        "observed": q["score"].get("observed", ""),
                        "wall_s": q["wall_s"],
                        "first_s": q["first_s"],
                        "peak_bytes": q["peak"],
                        "tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                        "answer": tr["answer"] or "",
                    }.items()
                }
            )
        requests.append(rec)
    csv_file("visit-residual-requests", requests)
    updates = unpack(ROOT / "updates.json.gz")
    names = ("objective", "source_ce", "margin", "rehearsal_ce", "donor_kl")
    losses = [
        {
            "step": r["step"],
            **dict(zip(names, r["losses"], strict=True)),
            "gradient_norm": r["gradient_norm"],
            "sampled_targets": r["sampled"],
        }
        for r in updates
    ]
    csv_file("visit-residual-optimization", losses)
    probes = unpack(ROOT / "training-interchange.json.gz")
    modes = ("donor", "learned", "paired_change")
    csv_file(
        "visit-residual-interchange",
        [
            {
                "source": row["source"],
                "id": row["id"],
                "peer": row["paired_id"],
                "target": row["target"],
                "foil": row["foil"],
                **{
                    mode + "_" + k: row[mode][k]
                    for mode in modes
                    for k in ("margin", "nll", "top_token")
                },
            }
            for row in probes
        ],
    )
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.65), layout="constrained")
    for name, label, color in zip(
        ("source_ce", "rehearsal_ce", "donor_kl"),
        ("New source CE", "Rehearsal CE", "Donor KL"),
        ("#176d60", "#795e90", "#ad693c"),
        strict=True,
    ):
        ax[0].plot(
            [j + 25 for j in range(0, len(losses), 50)],
            [
                mean(r[name] for r in losses[j : j + 50])
                for j in range(0, len(losses), 50)
            ],
            label=label,
            color=color,
            lw=1.4,
        )
    ax[0].set(xlabel="Native optimizer update", ylabel="Mean loss (50 updates)")
    ax[0].legend(frameon=False, fontsize=7)
    for j, (mode, label, color) in enumerate(
        zip(
            modes,
            ("Donor", "Learned", "Swapped"),
            ("#8e9794", "#176d60", "#795e90"),
            strict=True,
        )
    ):
        vals = sorted(z[mode]["margin"] for z in probes)
        ax[1].plot(
            vals, np.arange(1, len(vals) + 1) / len(vals), color=color, label=label
        )
    ax[1].axvline(0, color="#484d4a", ls=":", lw=0.8)
    ax[1].set(
        xlabel="Gold minus paired foil (logits)", ylabel="Fraction of 96 endpoints"
    )
    ax[1].legend(frameon=False, fontsize=7)
    fig.savefig(P / "visit-residual-training.svg", metadata={"Date": None})
    plt.close(fig)
    by = []
    for family, label in FAMILIES.items():
        left, right = (x["families"][family] for x in (a, b))
        by.append(
            [
                label,
                left["requests"],
                left["grounded"],
                right["grounded"],
                left["completed"],
                right["completed"],
            ]
        )
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.75), layout="constrained")
    for arm, z, color, label in zip(
        ARMS, (a, b), COLORS, ("Loop", "Base"), strict=True
    ):
        bytime = sorted(requests, key=lambda r: r[arm + "_wall_s"])
        times = [r[arm + "_wall_s"] for r in bytime]
        ax[0].step(
            times,
            np.cumsum([r[arm + "_success"] for r in bytime]),
            where="post",
            label=label,
            color=color,
        )
        ax[1].step(
            times,
            np.arange(1, len(times) + 1) / len(times),
            where="post",
            label=label,
            color=color,
        )
    for axis in ax:
        axis.set(xlabel="Per-request time allowance (s)", xscale="log")
        axis.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel("Correct by the deadline (of 112)")
    ax[1].set_ylabel("Fraction of requests finished")
    fig.savefig(P / "visit-residual-performance.svg", metadata={"Date": None})
    plt.close(fig)
    review = read(DOC / "visit-residual-reading.json")
    masked = []
    for family, label in (("effects", "Human"), ("same_version", "Wheel")):
        for metric in (
            "supported",
            "partial",
            "wrong",
            "absent",
            "unsupported_claim",
            "both_supported",
        ):
            masked.append(
                [
                    label,
                    metric.replace("_", " "),
                    *[review[k][family].get(metric, 0) for k in ARMS],
                ]
            )
    costs = []
    for key, label, scale, places in (
        ("grounded", "Correct or grounded", 1, 0),
        ("completed", "Completed answers", 1, 0),
        ("wall_s", "Total response time (s)", 1, 2),
        ("first_s", "Summed first-token time (s)", 1, 2),
        ("peak", "Peak MLX allocation (GiB)", 2**30, 3),
        ("output_tokens", "Generated tokens", 1, 0),
        ("opening_tps", "Physical uncached opening (tokens/s)", 1, 2),
        ("decode_tps", "Post-first decode (tokens/s)", 1, 2),
        ("effects_f1", "Mean human-question token F1", 1, 3),
    ):
        costs.append(
            [
                label,
                *[
                    f"{(z[key] if key in z else z['all'][key]) / scale:,.{places}f}"
                    for z in (a, b)
                ],
            ]
        )
    probe_table = []
    for mode, title in zip(
        modes,
        ("Frozen donor", "Learned correction", "Other question’s change"),
        strict=True,
    ):
        probe_table.append(
            [
                title,
                f"{mean(r[mode]['margin'] for r in probes):.3f}",
                f"{mean(r[mode]['nll'] for r in probes):.3f}",
                sum(r[mode]["margin"] > 0 for r in probes),
                sum(r[mode]["top_token"] == r["target"] for r in probes),
            ]
        )
    save(
        P / "visit-residual-paper.json",
        {
            "quality": by,
            "masked": masked,
            "cost": costs,
            "probe": probe_table,
            "whole": [[FAMILIES[k], a["whole"][k], b["whole"][k]] for k in FAMILIES],
            "human_p": f"{d['paired_p']:.5g}",
            "gates": ", ".join(k for k, v in d["gates"].items() if not v) or "none",
            "prior_lost": len(d["old_base_lost"]),
            "fit": fit,
            "decision": "qualified" if d["passed"] else "did not qualify",
            "net_correct": a["all"]["grounded"] - b["all"]["grounded"],
            "mean_learned_swap": f"{mean(r['learned']['margin'] - r['paired_change']['margin'] for r in probes):.3f}",
            "learned_gt_swap": sum(
                r["learned"]["margin"] > r["paired_change"]["margin"] for r in probes
            ),
        },
    )
    edition(P, "visit-residual")
    edition(P, "visit-residual", verify=True)


if __name__ == "__main__":
    build()
