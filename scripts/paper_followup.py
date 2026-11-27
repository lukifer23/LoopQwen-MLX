"""Publish real conversation continuation, current wheel choice and protected native fitting."""

import csv
from pathlib import Path

import matplotlib
import numpy as np
import typst
from pypdf import PdfReader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import read, save
from scripts.followup_learning import build as learning
from scripts.followup_reading import ARMS, DOC, ROOT, run

P = Path("paper")


def build():
    run("report")
    learning()
    d = read(ROOT / "decision.json")
    fit = read(ROOT / "fit.json")
    logs = read(DOC / "followup-learning.json")
    reviews = read(DOC / "followup-reading.json")
    arms = [d["arms"][a] for a in ARMS]
    with (P / "followup-requests.csv").open(newline="") as file:
        cases = list(csv.DictReader(file))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    colors = ["#21695d", "#766395"]
    families = (
        "human_first",
        "human_followup",
        "archive",
        "wheel",
        "agenda",
        "retention",
    )
    labels = ("First Q.", "Follow-up", "New wheel", "Prior wheel", "Schedule", "Lookup")
    for j, (a, arm, c, label) in enumerate(
        zip(ARMS, arms, colors, ("Trained loop", "Untouched Base"), strict=True)
    ):
        by = arm["families"]
        axes[0].barh(
            np.arange(len(families)) + (j - 0.5) * 0.38,
            [by[k]["grounded"] / by[k]["requests"] for k in families],
            height=0.35,
            color=c,
            label=label,
        )
        waits = sorted(float(row[a + "_wall_s"]) for row in cases)
        axes[1].step(
            waits,
            np.arange(1, len(waits) + 1) / len(waits),
            where="post",
            color=c,
            label=label,
        )
    axes[0].set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlim=(0, 1),
        xlabel="Fraction successful",
    )
    axes[0].invert_yaxis()
    axes[1].set(
        xscale="log", xlabel="Complete request time (s)", ylabel="Fraction of requests"
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(P / "followup-results.svg", metadata={"Date": None})
    plt.close(fig)
    overview = []
    for k, title in (
        ("human_first", "First human questions"),
        ("human_followup", "Human follow-ups"),
        ("archive", "Actual wheel and compatibility"),
        ("wheel", "Earlier release manifest"),
        ("agenda", "Earlier schedule"),
        ("file", "Prior file judgments"),
        ("inline", "Prior inline judgments"),
        ("retention", "Prior lookups"),
    ):
        g = [a["families"][k] for a in arms]
        overview.append(
            [
                title,
                str(g[0]["requests"]),
                *[str(r["grounded"]) for r in g],
                *[str(r["completed"]) for r in g],
            ]
        )
    costs = []
    for k, title, scale in (
        ("grounded", "Grounded or source-scored", 1),
        ("completed", "Completed answers", 1),
        ("wall_s", "Total request time (s)", 1),
        ("first_s", "Summed first-token wait (s)", 1),
        ("peak", "Peak native allocation (GiB)", 2**30),
        ("output_tokens", "Generated tokens", 1),
    ):
        digits = 3 if k == "peak" else 2 if k.endswith("_s") else 0
        costs.append([title, *[f"{a['all'][k] / scale:,.{digits}f}" for a in arms]])
    for k, title in (
        ("opening_tps", "Uncached opening tokens/s"),
        ("decode_tps", "Post-first generated tokens/s"),
    ):
        costs.append([title, *[f"{a[k]:,.2f}" for a in arms]])
    responses = []
    for a in ARMS:
        h = reviews[a]["masked_review"]
        o = reviews[a]["observations"]
        responses.append(
            [
                h.get("supported", 0),
                h.get("justified abstention", 0),
                h.get("partial", 0),
                h.get("wrong", 0),
                h.get("absent", 0),
                h["unsupported_claim"],
                o["human_followup"]["previous_repeated"],
                o["archive"]["current_read"],
                o["archive"]["previous_read"],
                o["archive"]["used_older_constraint"],
                o["archive"]["constraint"],
                o["archive"]["verdict"],
            ]
        )
    fields = (
        "Supported answer on masked sample",
        "Justified abstention",
        "Partial answer",
        "Incorrect answer",
        "No visible answer",
        "Unsupported factual claim",
        "Prior answer repeated, 24 follow-ups",
        "Current archive read, 24 requests",
        "Prior archive read, 24 requests",
        "Prior constraint repeated",
        "Exact current constraint",
        "Correct compatibility verdict",
    )
    residuals = []
    for kind in ("human_first", "human_followup", "action", "interpret"):
        ss = [w["samples"][kind] for w in logs["windows"] if kind in w["samples"]]
        n = sum(y["draws"] for y in ss)
        residuals.append(
            [
                kind,
                str(n),
                str(sum(y["gained"] for y in ss)),
                str(sum(y["lost"] for y in ss)),
                f"{sum(y['mean_delta'] * y['draws'] for y in ss) / n:.3f}",
            ]
        )
    net = sum(d["source_differences"].values())
    cap = (
        "qualified for the registered next stage"
        if d["passed"]
        else "missed the joint release threshold"
    )
    save(
        P / "followup-paper.json",
        {
            "abstract": f"On 48 human question presentations, the trained model gained {net:+d} successes against Base at the registered agreement threshold (paired p = {d['paired_p']:.4g}). Its average follow-up F1 was {arms[0]['mean_f1']['human_followup']:.3f} versus {arms[1]['mean_f1']['human_followup']:.3f}. It completed both compatibility decisions on {arms[0]['whole']['archive']}/12 actual project wheels versus {arms[1]['whole']['archive']}/12. The overall comparison {cap}.",
            "overview": overview,
            "costs": costs,
            "residuals": residuals,
            "masked": [
                [k, *map(str, v)]
                for k, v in zip(fields, zip(*responses, strict=True), strict=True)
            ],
            "whole": [
                [label, *[str(z["whole"][key]) for z in arms]]
                for key, label in (
                    ("human", "Both human questions"),
                    ("archive", "Both real-wheel decisions"),
                    ("wheel", "Both earlier package claims"),
                    ("agenda", "Both schedule decisions"),
                )
            ],
            "fit": f"{fit['trainable']:,} trainable parameters, {fit['targets']:,} new supervised token targets and {fit['protected_targets']:,} protected targets. We observed {logs['sampled_foils']:,} first-difference contrasts on {logs['unique_contrasted_views']} distinct training turns. {fit['projected']}/800 proposed steps needed projection. {sum(w['nonzero'] for w in logs['windows'])}/800 changed the stored weights. The largest measured protected-loss increase was {max(w['maximum_reference_increase'] for w in logs['windows']):.7f}.",
            "lost": str(len(d["lost"])),
            "failed": ", ".join(
                k.replace("_", " ") for k, v in d["gates"].items() if not v
            )
            or "none",
            "sha": fit["weights"],
        },
    )
    raw = typst.compile(str(P / "followup.typ"), timestamp=0)
    (P / "followup.pdf").write_bytes(raw)
    pages = PdfReader(P / "followup.pdf").pages
    assert len(pages) == 4 and all(len(page.extract_text()) > 400 for page in pages)
    print("Published", len(cases), "paired requests and", len(pages), "pages")


if __name__ == "__main__":
    build()
