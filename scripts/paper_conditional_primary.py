"""Publish the frozen 340-pair comparison, masked readings and source-level uncertainty."""

from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typst
from pypdf import PdfReader, PdfWriter

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_primary import ARMS, FRESH, configure, providers
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv
from scripts.assistance_uncertainty import (
    completed_pairs,
    cost_profile,
    joint,
    paired_sources,
)
from scripts.conditional_deliberation import account as channel_account
from scripts.conditional_reading import MASK, RATINGS, ROOT, SCORES
from scripts.conditional_reading import run as read_primary
from scripts.conditional_tool_transfer import build as tool_transfer
from scripts.paper_conditional_stability import build as repeat_review
from scripts.paper_integrated_assistance import COLORS, LABELS
from scripts.table_policy_reading import committed
from scripts.table_reasoning_reading import run as read_reasoning

P = Path("paper")
DOC = Path("docs/measurements")
OLD = Path("research/results/table-policy-gradient-01")
LABEL = {
    **LABELS,
    "changed_table": "Changed tables",
    "natural_mounted": "Wikipedia via SQLite",
    "natural_inline": "Wikipedia inline",
    "instruction": "New instructions",
}


def csv(label, rows):
    write_csv(P / ("conditional-primary-" + label + ".csv"), rows)


def get_outputs(record):
    return [t["token_ids"] for t in record["requests"][0]["trajectory"]["turns"]]


def render_charts(intervals, reads, costs):
    plt.rcParams.update({"svg.hashsalt": "conditional-primary-01", "font.size": 9})
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    for i, r in enumerate(intervals):
        ax.plot(
            [100 * r["resampled_low"], 100 * r["resampled_high"]],
            [i, i],
            color="#496b74",
            linewidth=1.6,
        )
        ax.scatter(100 * r["net_rate"], i, color=COLORS[0], s=24)
    ax.set_yticks(range(len(intervals)), [LABEL[r["family"]] for r in intervals])
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Loop − Base, percentage points")
    ax.set_title("A. Objective success by source family", loc="left", fontsize=11)
    reads = sorted(reads, key=lambda r: r["family"])
    y = np.arange(len(reads))
    for j, a in enumerate(ARMS):
        bx.barh(
            y + (j - 0.5) * 0.32,
            [100 * r[a + "_grounded_usable"] / r["items"] for r in reads],
            height=0.30,
            color=COLORS[j],
            label=("Loop", "Base")[j],
        )
    bx.set_yticks(y, [LABEL[r["family"]] for r in reads])
    bx.set_xlim(0, 100)
    bx.set_xlabel("Fully usable answers (%)")
    bx.set_title("B. Masked source reading", loc="left", fontsize=11)
    bx.legend(frameon=False)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.savefig(P / "conditional-primary-quality.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for a, c, title in zip(
        axes,
        ("first_s", "wall_s"),
        ("A. First token", "B. End-to-end request"),
        strict=True,
    ):
        for arm, color, label in zip(ARMS, COLORS, ("Loop", "Base"), strict=True):
            v = np.sort([r[arm + "_" + c] for r in costs])
            a.step(
                v,
                np.arange(1, len(v) + 1) / len(v),
                where="post",
                color=color,
                label=label,
            )
        a.set_xscale("log")
        a.set_xlabel("Measured seconds (log scale)")
        a.set_ylabel("Fraction of 340 requests")
        a.set_title(title, loc="left", fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.savefig(P / "conditional-primary-cost.svg", metadata={"Date": None})
    plt.close(fig)


def build():
    for receipt in (
        MASK,
        RATINGS,
        DOC / "conditional-reasoning-blinded.json",
        DOC / "conditional-reasoning-blinded-ratings.json",
    ):
        committed(receipt)
    read_primary("report")
    read_reasoning("report", True)
    mask, grades = read(MASK), read(RATINGS)
    reading, reasoning = (
        read(SCORES),
        read(DOC / "conditional-reasoning-source-reading.json"),
    )
    dec = read(ROOT / "decision.json")
    spec = read(ROOT / "registration.json")
    material = read(spec["material"])
    prev = {a: {r["id"]: r for r in unpack(OLD / (a + ".json.gz"))} for a in ARMS}
    streams = [unpack(ROOT / (a + ".json.gz")) for a in ARMS]
    rows = []
    calls = []
    prior = []
    ratings = []
    quality = defaultdict(Counter)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        configure(material, folder)
        for w, *pair in zip(material["rows"], *streams, strict=True):
            ident = w["id"]
            family = w["family"]
            row = {
                "id": ident,
                "family": family,
                "cluster": w.get("cluster", w.get("source", ident)),
                "tools_available": bool(providers(w, folder)),
                "opening_tokens": len(w["initial_ids"]),
                "opening_ids_sha256": fingerprint(w["initial_ids"]),
            }
            for arm, item in zip(ARMS, pair, strict=True):
                assert item["id"] == ident
                (q,) = item["requests"]
                tr = q["trajectory"]
                turns = tr["turns"]
                assert turns[0]["prompt_ids"] == w["initial_ids"]
                score = q["score"]
                ev = executions(tr)
                answer = tr["answer"] or ""
                values = {
                    "grounded": score.get("success", ""),
                    "completed": score["completed"],
                    "f1": score.get("f1", ""),
                    "visible": answer,
                    "status": tr["status"],
                    "wall_s": q["wall_s"],
                    "first_s": q["first_s"],
                    "peak_bytes": q["peak"],
                    "calls": len(ev),
                    "turns": len(turns),
                    "output_tokens": sum(len(t["token_ids"]) for t in turns),
                    "opening_work_tokens": sum(
                        len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0)
                        for t in turns
                    ),
                    "all_turn_first_s": sum(t["ttft_s"] for t in turns),
                    "decode_s": sum(t["seconds"] - t["ttft_s"] for t in turns),
                }
                row.update({arm + "_" + k: v for k, v in values.items()})
                for j, e in enumerate(ev, 1):
                    calls.append(
                        {
                            "id": ident,
                            "family": family,
                            "arm": arm,
                            "ordinal": j,
                            "name": e["call"]["name"],
                            "arguments": e["call"]["arguments"],
                            "observation": e["observation"],
                        }
                    )
                if family not in FRESH:
                    old = prev[arm][ident]
                    oq = old["requests"][0]
                    assert (
                        oq["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                    )
                    prior.append(
                        {
                            "id": ident,
                            "arm": arm,
                            "family": family,
                            "same_tokens": get_outputs(item) == get_outputs(old),
                            "old_success": oq["score"].get("success", ""),
                            "new_success": values["grounded"],
                            "old_wall_s": oq["wall_s"],
                            "new_wall_s": q["wall_s"],
                        }
                    )
                if ident in mask:
                    offset = mask[ident]["replies"].index(answer)
                    grade = grades[ident]["replies"][offset]
                    usable = grade["rating"] == "fully usable" and (
                        family != "natural_mounted" or score["supported_read"]
                    )
                    ratings.append(
                        {
                            "id": ident,
                            "family": family,
                            "arm": arm,
                            "rating": grade["rating"],
                            "grounded_usable": usable,
                            "unsupported": grade["unsupported"],
                            "preferred": grades[ident]["preferred"] == offset,
                            "rationale": grades[ident]["reason"],
                        }
                    )
                    quality[family][arm + "_grounded_usable"] += usable
                    quality[family][arm + "_unsupported"] += grade["unsupported"]
                    quality[family][arm + "_preferred"] += (
                        grades[ident]["preferred"] == offset
                    )
            rows.append(row)
    assert len(rows) == 340 and len(prior) == 536 and len(ratings) == 312
    csv("requests", rows)
    csv("real-calls", calls)
    csv("previous", prior)
    csv("reading", ratings)
    legacy = [r for r in rows if r["family"] not in FRESH]
    obj = [r for r in legacy if r["family"] != "open"]
    assert len(obj) == 256
    by_family = []
    for f in sorted({r["family"] for r in obj}):
        group = [r for r in obj if r["family"] == f]
        by_family.append(
            [
                LABEL[f],
                len(group),
                *[sum(r[a + "_grounded"] for r in group) for a in ARMS],
            ]
        )
    qs = [
        {"family": f, "items": sum(r["family"] == f for r in rows), **v}
        for f, v in sorted(quality.items())
    ]
    assert all(
        r[a + "_grounded_usable"] == reading["arms"][a][r["family"]]["grounded_usable"]
        for r in qs
        for a in ARMS
    )
    intervals = paired_sources(legacy)
    completion = completed_pairs(legacy)
    csv("source-intervals", intervals)
    csv("cost-distribution", cost_profile(rows))
    csv("completion", completion)
    diagnostics = tool_transfer()
    render_charts(intervals, qs, rows)
    unrolled = {}
    for group, gates in (
        ("product", dec["gates"]),
        ("retention", dec["retention"]["gates"]),
        ("masked", reading["gates"]),
    ):
        unrolled.update({group + "/" + k: bool(v) for k, v in gates.items()})
    differences = [
        sum(r[a + "_grounded"] and not r[b + "_grounded"] for r in obj)
        for a, b in (ARMS, ARMS[::-1])
    ]

    def results(rr):
        return [sum(r[a + "_grounded"] for r in rr) for a in ARMS]

    decode = []
    for arm in ARMS:
        count = sum(r[arm + "_output_tokens"] - r[arm + "_turns"] for r in rows)
        seconds = sum(r[arm + "_decode_s"] for r in rows)
        assert isclose(count / seconds, dec["cost"][arm]["decode_tps"])
        decode.append((count, seconds / count))
    (nc, tc), (nb, tb) = decode
    length_s = (nb - nc) * (tc + tb) / 2
    rate_s = (tb - tc) * (nb + nc) / 2
    assert isclose(length_s + rate_s, nb * tb - nc * tc, abs_tol=1e-6)
    overlap = read(DOC / "conditional-source-overlap.json")
    assert not any(rec["whole_field"] for rec in overlap["fields"])
    summary = {
        "environment": read(ROOT / "environment.json"),
        "generation_channels": channel_account(ROOT, streams),
        "tool_diagnostics": diagnostics,
        "decode_savings": {
            "length": length_s,
            "rate": rate_s,
            "total": length_s + rate_s,
        },
        "warm_wall": [sum(r[a + "_wall_s"] for r in rows[1:]) for a in ARMS],
        "warm_first": [sum(r[a + "_first_s"] for r in rows[1:]) for a in ARMS],
        "overlap": {
            "fields": overlap["field_count"],
            "occurrences": overlap["training_word_occurrences"],
            "maximum_run": max(
                rec.get("matched_words", 0) for rec in overlap["fields"]
            ),
        },
        "automatic": dec["passed"],
        "masked": reading["passed"],
        "eligible": dec["passed"] and reading["passed"],
        "gates": unrolled,
        "objective": results(obj),
        "gains_losses": differences,
        "families": by_family,
        "joint": joint(legacy),
        "tools": results([r for r in obj if r["tools_available"]]),
        "ordinary": results([r for r in obj if not r["tools_available"]]),
        "both_finished": next(
            v
            for v in completion
            if v["family"] == "all" and v["completion"] == "Both completed"
        ),
        "reading": qs,
        "whole": reading["whole"],
        "reasoning": reasoning,
        "f1": {a: dec["retention"]["arms"][a]["f1"] for a in ARMS},
        "whole_tables": dec["retention"]["whole_table_counts"],
        "fresh": dec["fresh"],
        "costs": dec["cost"],
        "prior_base_identical": sum(
            r["same_tokens"] for r in prior if r["arm"] == "control"
        ),
        "reading_losses": reading["usable_losses"],
        "source_sha256": digest(ROOT / "checksums.json"),
    }
    save(P / "conditional-primary.json", summary)
    pdf = P / "conditional-primary.pdf"
    pdf.write_bytes(
        typst.compile(str(P / "conditional-primary.typ"), root=".", timestamp=0)
    )
    repeat_review()
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Title": "Does a trained recurrent Qwen improve assistance across tasks?",
            "/Author": "LoopQwen-MLX",
        }
    )
    for title, path in (
        ("Direct paired evaluation and masked source readings", pdf),
        ("Item stability and repeat grading", P / "conditional-stability.pdf"),
        (
            "Training, architecture and earlier comparisons",
            P / "loopqwen-conditional-study.pdf",
        ),
    ):
        doc = PdfReader(path)
        assert all(len(page.extract_text()) > 160 for page in doc.pages)
        writer.append(doc, outline_item=title)
    with (P / "loopqwen-conditional-primary-study.pdf").open("wb") as out:
        writer.write(out)


if __name__ == "__main__":
    build()
