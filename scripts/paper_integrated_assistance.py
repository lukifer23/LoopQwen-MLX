"""Reconstruct quality, tool execution, merge diagnostics and costs from the sealed pair."""

from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assemble_evidence_paper import edition
from scripts.assistance_deliberation import write_csv
from scripts.assistance_source_overlap import build as source_overlap
from scripts.assistance_reasoning_exposure import (
    DEST as THINKING_RECEIPT,
    observed as reasoning_exposure,
)
from scripts.assistance_uncertainty import (
    completed_pairs,
    cost_profile,
    joint,
    open_p,
    paired_sources,
)
from scripts.integrated_assistance_reading import RATINGS, ROOT
from scripts.integrated_assistance_reading import run as reading
from scripts.integrated_training_exposure import build as exposure

P = Path("paper")
ARMS = ("candidate", "control")
COLORS = ("#176859", "#765487")
LABELS = {
    "database": "New database questions",
    "open": "New open assistance",
    "archive": "Python compatibility",
    "dependency": "Declared dependencies",
    "same_version": "Fixed-version changes",
    "human_first": "First human questions",
    "human_followup": "Human follow-ups",
    "effects": "Human causal questions",
    "agenda": "Schedules",
    "wheel": "Earlier manifests",
    "retention": "Earlier lookups",
    "file": "File judgments",
    "inline": "Inline judgments",
}


def store(stem, rows):
    write_csv(P / ("assistance-" + stem + ".csv"), rows)


def figure(fig, stem):
    fig.savefig(P / ("assistance-" + stem + ".svg"), metadata={"Date": None})
    plt.close(fig)


def train():
    spec, fit = (read(ROOT / name) for name in ("registration.json", "fit.json"))
    steps = unpack(ROOT / "optimizer-steps.json.gz")
    views = read(spec["training"])["examples"]
    history = []
    for step in steps:
        for j, (index, metrics) in enumerate(
            zip(step["views"], step["components"], strict=True)
        ):
            w = views[index]
            group = w["kind"] if w["group"] == "database" else w["group"]
            second, first, rank = metrics
            history.append(
                {
                    "step": step["step"],
                    "group": group,
                    "view": w["id"],
                    "targets": len(step["positions"][j]),
                    "second_nll": second,
                    "first_nll": first,
                    "gold_logp_gain": first - second,
                    "ranking": rank,
                    "gradient_norm": step["gradient_norm"],
                }
            )
    store("optimizer", history)
    merged = read(ROOT / "task-merge.json")
    projections = []
    for name, row in sorted(merged["projections"].items()):
        projections.append(
            dict(
                projection=name,
                elements=row["elements"],
                opposite_signs=row["opposite_signs"],
                nonzero=row["nonzero"],
                **{
                    "contributors_" + str(n): c
                    for n, c in enumerate(row["participation"])
                },
                **{
                    "source_" + str(n) + "_retained": c
                    for n, c in enumerate(row["retained_per_source"])
                },
            )
        )
    store("merge", projections)
    points = []
    for w in unpack(ROOT / "merged-training-visits.json.gz"):
        for j, at in enumerate(w["positions"]):
            points.append(
                dict(
                    group=w["group"],
                    view=w["view"],
                    position=at,
                    target=w["target"][j],
                    **{
                        a + "_" + m: w[a][m][j]
                        for a in ("first", "second")
                        for m in ("top", "nll")
                    },
                )
            )
    store("visits", points)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 2.65), layout="constrained")
    blocks = defaultdict(list)
    for row in history:
        blocks[(row["step"] - 1) // 50].append(row)
    for k, label, col in (
        ("first_nll", "First visit", COLORS[1]),
        ("second_nll", "Second visit", COLORS[0]),
    ):
        left.plot(
            [50 * j + 25 for j in blocks],
            [mean(r[k] for r in rows) for rows in blocks.values()],
            label=label,
            color=col,
        )
    left.set(xlabel="Native optimizer update", ylabel="Gold-token NLL (50-update mean)")
    left.legend(frameon=False, fontsize=8)
    y = np.arange(len(projections))
    names = [r["projection"].split(".")[-2].replace("_proj", "") for r in projections]
    for key, label, color in (
        ("nonzero", "Active after merge", COLORS[0]),
        ("opposite_signs", "Opposing signs", COLORS[1]),
    ):
        right.barh(
            y + (-0.18 if key == "nonzero" else 0.18),
            [100 * r[key] / r["elements"] for r in projections],
            height=0.34,
            label=label,
            color=color,
        )
    right.set(yticks=y, yticklabels=names, xlabel="Percent of projection coordinates")
    right.legend(frameon=False, fontsize=7)
    figure(fig, "learning")
    return {
        "targets": fit["targets"],
        "fit_seconds": round(fit["optimizer_seconds"], 2),
        "positions": len(points),
        "first_hits": sum(r["first_top"] == r["target"] for r in points),
        "second_hits": sum(r["second_top"] == r["target"] for r in points),
        "first_nll": f"{mean(r['first_nll'] for r in points):.3f}",
        "second_nll": f"{mean(r['second_nll'] for r in points):.3f}",
        "merged": merged["compiled"],
        "opposed": sum(r["opposite_signs"] for r in projections),
        "coordinates": sum(r["elements"] for r in projections),
    }


def evaluation():
    d = read(ROOT / "decision.json")
    spec = read(ROOT / "registration.json")
    panel = read(spec["material"])["rows"]
    runs = [unpack(ROOT / (a + ".json.gz")) for a in ARMS]
    records, sql, phases = [], [], []
    for w, *pair in zip(panel, *runs, strict=True):
        row = {"id": w["id"], "family": w["family"], "cluster": w["cluster"]}
        for a, case in zip(ARMS, pair, strict=True):
            assert case["id"] == w["id"]
            (q,) = case["requests"]
            tr = q["trajectory"]
            s = q["score"]
            row.update(
                {
                    a + "_" + k: v
                    for k, v in {
                        "grounded": s.get("success", ""),
                        "completed": s["completed"],
                        "status": tr["status"],
                        "wall_s": q["wall_s"],
                        "first_s": q["first_s"],
                        "peak_bytes": q["peak"],
                        "tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                        "answer": tr["answer"] or "",
                        "f1": s.get("f1", ""),
                    }.items()
                }
            )
            turns = tr["turns"]
            opening = sum(t["ttft_s"] for t in turns)
            decode = sum(t["seconds"] - t["ttft_s"] for t in turns)
            overhead = q["wall_s"] - opening - decode
            assert min(opening, decode, overhead) >= -1e-6
            phases.append(
                {
                    "id": w["id"],
                    "family": w["family"],
                    "arm": a,
                    "turns": len(turns),
                    "uncached_prompt_tokens": sum(
                        len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0)
                        for t in turns
                    ),
                    "post_first_output_tokens": sum(
                        len(t["token_ids"]) - 1 for t in turns
                    ),
                    "all_turn_opening_s": opening,
                    "all_post_first_decode_s": decode,
                    "outside_generation_s": overhead,
                    "request_wall_s": q["wall_s"],
                    "first_turn_generation_s": turns[0]["seconds"],
                }
            )
            if w["family"] == "database":
                from loopqwen.research.native_function_study import executions

                calls = executions(tr)
                sql.append(
                    {
                        "id": w["id"],
                        "database": w["source"],
                        "question": w["questions"][0],
                        "arm": a,
                        "reference_query": w["sql"],
                        "expected": str(w["expected_value"]),
                        "query_calls": s["observed_queries"],
                        "schema_calls": s["schema_calls"],
                        "correct_original": s["original_result"],
                        "alternative_matches": s["semantic"]["matches"],
                        "alternatives": s["semantic"]["instances"],
                        "semantic_pass": s["semantic"]["passed"],
                        "grounded": s["success"],
                        "generated_queries": "\n".join(
                            str(e["call"].get("arguments", {}).get("statement", ""))
                            for e in calls
                            if e["call"].get("name") == "database_query"
                        ),
                        "answer": tr["answer"] or "",
                    }
                )
        records.append(row)
    store("requests", records)
    store("latency-phases", phases)
    common = completed_pairs(records)
    store("common-completion", common)
    (both,) = [
        g
        for g in common
        if g["family"] == "all" and g["completion"] == "Both completed"
    ]
    overlap = source_overlap()["original_utterance"]["all"]
    assert overlap["exact"] == 1
    # Omit the entire independently counted database, preserving both questions in every other pair.
    duplicate = [
        r
        for r in records
        if r["family"] == "database" and r["cluster"] == "course_teach"
    ]
    assert len(duplicate) == 2
    remaining = [r for r in records if r["family"] == "database" and r not in duplicate]
    (sensitivity,) = paired_sources(remaining)
    assert sensitivity["requests"] == 22 and sensitivity["registered_groups"] == 11
    store("sql-utterance-sensitivity", [sensitivity])
    by_source = paired_sources(records)
    costs_by_item = cost_profile(records)
    store("source-uncertainty", by_source)
    store("cost-percentiles", costs_by_item)
    store("sql", sql)
    a, b = [d["arms"][k] for k in ARMS]
    family = []
    for group, label in LABELS.items():
        subset = [w for w in records if w["family"] == group]
        if group == "open":
            family.append(
                [label, len(subset), "—", "—", a["open_completed"], b["open_completed"]]
            )
            continue
        ma, mb = [z["families"][group] for z in (a, b)]
        family.append(
            [
                label,
                len(subset),
                ma["grounded"],
                mb["grounded"],
                ma["completed"],
                mb["completed"],
            ]
        )
    gains = sum(
        r["candidate_grounded"] is True and r["control_grounded"] is False
        for r in records
    )
    losses = sum(
        r["candidate_grounded"] is False and r["control_grounded"] is True
        for r in records
    )
    assert gains - losses == a["scored"]["grounded"] - b["scored"][
        "grounded"
    ] and losses == len(d["base_lost"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 2.6), layout="constrained")
    for arm, label, color in zip(
        ARMS, ("Trained loop", "Untouched Base"), COLORS, strict=True
    ):
        rows = sorted(
            (r for r in records if r["family"] != "open"),
            key=lambda r: r[arm + "_wall_s"],
        )
        left.step(
            [r[arm + "_wall_s"] for r in rows],
            np.cumsum([r[arm + "_grounded"] for r in rows]),
            where="post",
            color=color,
            label=label,
        )
        ss = [r for r in sql if r["arm"] == arm]
        vals = [
            sum(r["schema_calls"] > 0 for r in ss),
            sum(r["query_calls"] > 0 for r in ss),
            sum(r["correct_original"] for r in ss),
            sum(r["semantic_pass"] for r in ss),
            sum(r["grounded"] for r in ss),
        ]
        right.plot(range(5), vals, marker="o", color=color, label=label)
    left.set(
        xscale="log",
        xlabel="Allowed time per request (s)",
        ylabel="Correct objective requests (of 208)",
    )
    right.set(
        xticks=range(5),
        xticklabels=["Schema", "Query", "Original", "Suite", "Final"],
        ylabel="Database requests (of 24)",
        ylim=(-0.8, 25),
    )
    left.legend(frameon=False, fontsize=8)
    figure(fig, "outcomes")
    costs = []
    for metric, label, scale, nd in (
        ("wall_s", "Total time (s)", 1, 2),
        ("first_s", "Summed first-token time (s)", 1, 2),
        ("peak", "Peak native allocation (GiB)", 2**30, 3),
        ("opening_tps", "Uncached opening (tokens/s)", 1, 2),
        ("decode_tps", "Post-first generation (tokens/s)", 1, 2),
        ("output_tokens", "Generated tokens", 1, 0),
    ):
        costs.append(
            [label, *[f"{z['all_cost'][metric] / scale:,.{nd}f}" for z in (a, b)]]
        )
    review = read("docs/measurements/integrated-assistance-reading.json")
    grades = [
        [LABELS[family], k, *[review["arms"][arm][family].get(k, 0) for arm in ARMS]]
        for family in ("open", "database")
        for k in (
            "fully usable",
            "partial",
            "wrong",
            "absent",
            "unsupported",
            "win",
            "tie",
        )
    ]
    return {
        "family": family,
        "intervals": [
            [
                LABELS[r["family"]],
                r["registered_groups"],
                r["gains"],
                r["losses"],
                f"{100 * r['net_rate']:+.1f}",
                f"{100 * r['resampled_low']:+.1f} to {100 * r['resampled_high']:+.1f}",
                f"{r['holm_p']:.3g}",
            ]
            for r in by_source
        ],
        "percentiles": [
            [
                r["metric"],
                r["loop_lower"],
                f"{r['loop_p50']:.3f}",
                f"{r['base_p50']:.3f}",
                f"{r['loop_p95']:.3f}",
                f"{r['base_p95']:.3f}",
            ]
            for r in costs_by_item
        ],
        "phase_costs": [
            [
                label,
                *[
                    f"{sum(r[field] for r in phases if r['arm'] == arm):,.2f}"
                    for arm in ARMS
                ],
            ]
            for field, label in (
                ("all_turn_opening_s", "Opening through first token on every turn"),
                ("all_post_first_decode_s", "Decoding after the first token"),
                ("outside_generation_s", "Time outside timed generation"),
            )
        ],
        "sql_overlap_net": sensitivity["gains"] - sensitivity["losses"],
        "sql_overlap_p": f"{sensitivity['exact_group_p']:.4g}",
        "both_finished": both,
        "joint": joint(records),
        "preference_p": f"{open_p(*[review['arms'][arm]['open'] for arm in ARMS]):.5g}",
        "costs": costs,
        "grades": grades,
        "correct": [z["scored"]["grounded"] for z in (a, b)],
        "open_win": [review["arms"][arm]["open"].get("win", 0) for arm in ARMS],
        "whole_sql": [z["whole"]["database"] for z in (a, b)],
        "sql": [z["families"]["database"]["grounded"] for z in (a, b)],
        "f1": [
            [LABELS[g], *[f"{z['f1'][g]:.3f}" for z in (a, b)]]
            for g in ("human_first", "human_followup", "effects")
        ],
        "gains": gains,
        "losses": losses,
        "prior_lost": len(d["prior_lost"]),
        "prior_total": d["prior_successes"],
        "database_p": f"{d['database_p']:.5g}",
        "gates": [
            [k.replace("_", " "), "pass" if v else "miss"]
            for k, v in {**d["gates"], "blinded_review": review["reading_gate"]}.items()
        ],
        "qualified": d["passed"] and review["reading_gate"],
    }


def build():
    assert RATINGS.is_file()
    reading("report")
    exposure(ROOT)
    thinking = reasoning_exposure(ROOT)
    assert thinking == read(THINKING_RECEIPT)
    plt.rcParams.update(
        {
            "svg.hashsalt": "integrated-assistance-01",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    save(
        P / "integrated-assistance-paper.json",
        {
            **train(),
            **evaluation(),
            "reasoning_targets": thinking["totals"]["thinking_text"],
            "reasoning_draws": thinking["totals"]["draws_with_thinking_target"],
            "archive": digest(ROOT / "checksums.json"),
        },
    )
    edition(P, "integrated-assistance")
    edition(P, "integrated-assistance", verify=True)


if __name__ == "__main__":
    build()
