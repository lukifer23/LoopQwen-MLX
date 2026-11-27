"""Build the complete paired four-layer evaluation and link both native training studies."""

from collections import Counter, defaultdict
from io import BytesIO
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
from loopqwen.research.native_function_study import executions
from loopqwen.research.table_policy_primary import mounted, providers
from loopqwen.research.table_worlds import SOURCE, query_results
from scripts.assistance_deliberation import write_csv
from scripts.assistance_uncertainty import (
    completed_pairs,
    cost_profile,
    joint,
    paired_sources,
)
from scripts.paper_integrated_assistance import ARMS, COLORS, LABELS
from scripts.table_policy_reading import (
    MASK,
    RATINGS,
    ROOT,
    SCORES,
    committed,
)
from scripts.table_policy_reading import (
    run as reading,
)

P = Path("paper")
LABEL = {**LABELS, "changed_table": "Changed source tables"}


def table_csv(name, values):
    write_csv(P / ("table-policy-" + name + ".csv"), values)


def build():
    committed(RATINGS)
    committed(MASK)
    reading("report")
    d = read(ROOT / "decision.json")
    m = read(SCORES)
    s = read(ROOT / "registration.json")
    panel = read(s["material"])["rows"]
    tables = {w["id"]: w for w in read(SOURCE)["cohorts"]["dev"]}
    evaluations = [unpack(ROOT / (side + ".json.gz")) for side in ARMS]
    rows = []
    sql = []
    costs = []
    changed = defaultdict(Counter)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        mounted(read(s["material"]), folder)
        for w, *pair in zip(panel, *evaluations, strict=True):
            rec = {
                "id": w["id"],
                "family": w["family"],
                "tools_available": bool(providers(w, folder)),
                "cluster": w["cluster"],
                "opening_tokens": len(w["initial_ids"]),
                "initial_ids": fingerprint(w["initial_ids"]),
            }
            for side, case in zip(ARMS, pair, strict=True):
                assert case["id"] == w["id"]
                (q,) = case["requests"]
                tr = q["trajectory"]
                turns = tr["turns"]
                sc = q["score"]
                assert turns[0]["prompt_ids"] == w["initial_ids"]
                calls = executions(tr)
                rec.update(
                    {
                        side + "_" + k: v
                        for k, v in {
                            "grounded": sc.get("success", ""),
                            "completed": sc["completed"],
                            "status": tr["status"],
                            "visible": tr["answer"] or "",
                            "f1": sc.get("f1", ""),
                            "calls": len(calls),
                            "tokens": sum(len(t["token_ids"]) for t in turns),
                            "wall_s": q["wall_s"],
                            "first_s": q["first_s"],
                            "peak_bytes": q["peak"],
                        }.items()
                    }
                )
                opening = sum(t["ttft_s"] for t in turns)
                decode = sum(t["seconds"] - t["ttft_s"] for t in turns)
                costs.append(
                    {
                        "id": w["id"],
                        "arm": side,
                        "family": w["family"],
                        "tools_available": rec["tools_available"],
                        "opening_s": opening,
                        "decode_s": decode,
                        "other_s": q["wall_s"] - opening - decode,
                        "wall_s": q["wall_s"],
                    }
                )
                if w["family"] != "changed_table":
                    continue
                group = changed[side]
                group["any_tool"] += bool(calls)
                group["read"] += sc["observed"]
                group["completed"] += sc["completed"]
                group["correct"] += sc["success"]
                any_scalar = False
                for j, ev in enumerate(calls):
                    call = ev["call"]
                    if call["name"] != "database_query":
                        continue
                    statement = call["arguments"].get("statement")
                    if not isinstance(statement, str):
                        continue
                    world = tables[w["row_id"]]
                    replay = query_results(statement, world, folder)
                    index = [v["world"] for v in world["versions"]].index(w["world"])
                    row = replay[index]
                    if "observation" in row:
                        assert row["observation"] == ev["observation"]
                    any_scalar |= row["correct"]
                    sql.append(
                        {
                            "id": w["id"],
                            "arm": side,
                            "tool_number": j + 1,
                            "world": w["world"],
                            "question": world["question"],
                            "reference": w["gold"],
                            "statement": statement,
                            "scalar_correct_presented": row["correct"],
                            "scalar_correct_opposite": replay[1 - index]["correct"],
                            "read_presented": row["used"],
                            "observed": str(ev["observation"]),
                        }
                    )
                group["correct_scalar"] += any_scalar
            rows.append(rec)
    assert len(rows) == 268
    table_csv("requests", rows)
    table_csv("timing", costs)
    if sql:
        table_csv("executed-sql", sql)
    intervals = paired_sources(rows)
    table_csv("source-intervals", intervals)
    profile = cost_profile(rows)
    table_csv("cost-distributions", profile)
    comp = completed_pairs(rows)
    table_csv("completion", comp)
    obj = [r for r in rows if r["family"] != "open"]
    totals = [sum(r[a + "_grounded"] for r in obj) for a in ARMS]
    wins, losses = (
        sum(bool(r[a + "_grounded"]) and not r[b + "_grounded"] for r in obj)
        for a, b in (ARMS, ARMS[::-1])
    )
    assert losses == len(d["base_lost"]) and wins - losses == totals[0] - totals[1]
    frows = []
    for f, rr in sorted(
        (
            (k, [r for r in rows if r["family"] == k])
            for k in {r["family"] for r in rows}
        ),
        key=lambda kv: LABEL[kv[0]],
    ):
        frows.append(
            [
                LABEL[f],
                len(rr),
                *[
                    sum(r[a + "_grounded"] is True for r in rr) if f != "open" else "—"
                    for a in ARMS
                ],
                *[sum(r[a + "_completed"] for r in rr) for a in ARMS],
            ]
        )
    costs_all = [d["arms"][a]["all_cost"] for a in ARMS]
    costrows = [
        [label, *[f"{v[k] * scale:,.2f}" for v in costs_all]]
        for k, label, scale in (
            ("wall_s", "All requests (s)", 1),
            ("first_s", "Summed first-token time (s)", 1),
            ("opening_tps", "Opening rate (tokens/s)", 1),
            ("decode_tps", "Post-first decode rate (tokens/s)", 1),
            ("peak", "Maximum MLX allocation (GiB)", 2**-30),
            ("output_tokens", "Emitted tokens", 1),
        )
    ]
    masked = []
    for f in ("changed_table", "database", "open"):
        for a, label in zip(ARMS, ("Loop", "Base"), strict=True):
            v = m["arms"][a][f]
            masked.append(
                [
                    LABEL[f],
                    label,
                    *[
                        v.get(k, 0)
                        for k in (
                            "fully usable",
                            "partial",
                            "wrong",
                            "absent",
                            "unsupported",
                            "win",
                        )
                    ],
                ]
            )
    masks = read(MASK)
    ratings = read(RATINGS)
    examples = []
    for f in ("changed_table", "database", "open"):
        k = min(
            [w["id"] for w in panel if w["family"] == f],
            key=lambda k: fingerprint([19498, "paper-example", k]),
        )
        examples.append(
            {
                "id": k,
                "question": masks[k]["request"],
                "reason": ratings[k]["reason"],
                "reference": masks[k]["reference"],
                "replies": [
                    next(r[a + "_visible"] for r in rows if r["id"] == k) for a in ARMS
                ],
            }
        )
    for e in examples:
        e["slots"] = [
            masks[e["id"]]["replies"].index(answer) + 1 for answer in e["replies"]
        ]
        e["question"] = e["question"][:350] + ("…" if len(e["question"]) > 350 else "")
        e["replies"] = [
            answer[:210] + ("…" if len(answer) > 210 else "") for answer in e["replies"]
        ]
    # The earlier two-visit policy used the same 220 requests and identical user token IDs.
    from scripts.verify_runtime_wheel import trace

    preceding = Path("research/results/integrated-assistance-01")
    previous = [
        {r["id"]: r["requests"][0] for r in unpack(preceding / (side + ".json.gz"))}
        for side in ARMS
    ]
    transitions = []
    for w, current_loop, current_base in zip(panel, *evaluations, strict=True):
        k = w["id"]
        if k not in previous[0]:
            continue
        a, b = [run["requests"][0] for run in (current_loop, current_base)]
        p, q = [side[k] for side in previous]
        assert all(
            item["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
            for item in (a, b, p, q)
        )
        transitions.append(
            {
                "id": k,
                "family": w["family"],
                "identical_base_trajectory": trace(b["trajectory"])
                == trace(q["trajectory"]),
                "old_loop": p["score"].get("success", ""),
                "current_loop": a["score"].get("success", ""),
                "old_base": q["score"].get("success", ""),
                "current_base": b["score"].get("success", ""),
            }
        )
    assert len(transitions) == 220
    table_csv("previous-policy", transitions)
    comparative = {
        "identical": sum(r["identical_base_trajectory"] for r in transitions),
        "new": sum(
            r["current_loop"] is True and r["old_loop"] is False for r in transitions
        ),
        "lost": sum(
            r["current_loop"] is False and r["old_loop"] is True for r in transitions
        ),
        "scores": [
            sum(r[k] is True for r in transitions) for k in ("current_loop", "old_loop")
        ],
    }
    practical = []
    for z, label in ((True, "Tools supplied"), (False, "Plain questions")):
        eligible = {r["id"] for r in rows if r["tools_available"] == z}
        rr = [r for r in obj if r["id"] in eligible]
        practical.append(
            [
                label,
                len(rr),
                *[sum(r[a + "_grounded"] for r in rr) for a in ARMS],
                *[round(sum(r[a + "_wall_s"] for r in rr), 2) for a in ARMS],
            ]
        )
    summary = {
        "archive": digest(ROOT / "checksums.json"),
        "policy": read(ROOT / "training-parent.json")["weights"],
        "passed": d["passed"] and m["reading_gate"],
        "totals": totals,
        "previous": comparative,
        "wins": wins,
        "losses": losses,
        "family": frows,
        "whole": d["whole_table_counts"],
        "p": d["whole_table_p"],
        "semantic_whole": m["whole_tables"],
        "semantic_p": m["p"],
        "versions": [[k.title(), *v] for k, v in d["table_versions"].items()],
        "masked": masked,
        "lost_semantic": m["lost_individual"],
        "cost": costrows,
        "prior_lost": len(d["prior_lost"]),
        "prior_n": d["prior_successes"],
        "toolplain": practical,
        "gates": [
            [k.replace("_", " "), "Pass" if v else "Miss"]
            for k, v in {
                **d["gates"],
                **{"masked " + k: v for k, v in m["gates"].items()},
            }.items()
        ],
        "f1": [
            [LABEL[k], *[round(d["arms"][a]["f1"][k], 3) for a in ARMS]]
            for k in d["arms"]["candidate"]["f1"]
        ],
        "execution": [
            [label, *[changed[a][k] for a in ARMS]]
            for k, label in (
                ("any_tool", "Received a callback"),
                ("read", "Read the relevant table"),
                ("correct_scalar", "Query returned the exact scalar"),
                ("completed", "Delivered a complete answer"),
                ("correct", "Delivered a grounded correct answer"),
            )
        ],
        "examples": examples,
        "intervals": [
            [
                LABEL[r["family"]],
                r["registered_groups"],
                r["gains"],
                r["losses"],
                f"{100 * r['net_rate']:+.1f}",
                f"{100 * r['resampled_low']:.1f} to {100 * r['resampled_high']:.1f}",
                f"{r['holm_p']:.3f}",
            ]
            for r in intervals
        ],
        "joint": joint(rows),
    }
    save(P / "table-policy-paper.json", summary)
    plt.rcParams.update(
        {
            "svg.hashsalt": "table-policy-comparison",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained")
    for a, color, name in zip(
        ARMS, COLORS, ("Trained loop", "Untouched Base"), strict=True
    ):
        rr = sorted(obj, key=lambda r: r[a + "_wall_s"])
        axes[0].step(
            [r[a + "_wall_s"] for r in rr],
            np.cumsum([r[a + "_grounded"] for r in rr]),
            color=color,
            label=name,
            where="post",
        )
        arr = np.sort([r[a + "_first_s"] for r in rows])
        axes[1].plot(arr, np.arange(1, len(arr) + 1) / len(arr), color=color)
    axes[0].set(
        xlabel="Individual request time (s)",
        ylabel="Grounded answers delivered by this time",
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        xlabel="First-token latency (s, logarithmic)",
        ylabel="Cumulative fraction of requests",
        xscale="log",
    )
    fig.savefig(P / "table-policy-performance.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.35), layout="constrained")
    y = np.arange(len(intervals))
    ax.axvline(0, color="#777777", lw=0.8)
    for i, r in enumerate(intervals):
        lo, mid, hi = (
            100 * r[k] for k in ("resampled_low", "net_rate", "resampled_high")
        )
        ax.plot([lo, hi], [i, i], color=COLORS[0])
        ax.scatter(mid, i, color=COLORS[0], s=18)
    ax.set(
        yticks=y,
        yticklabels=[LABEL[r["family"]] for r in intervals],
        xlabel="Loop minus Base in grounded answers (percentage points)",
    )
    fig.savefig(P / "table-policy-source-effects.svg", metadata={"Date": None})
    plt.close(fig)
    new = typst.compile(str(P / "table-policy.typ"), timestamp=0)
    (P / "table-policy.pdf").write_bytes(new)
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Title": "A four-layer loop for general assistance and changed-source tool use",
            "/Author": "LoopQwen-MLX",
        }
    )
    previous = P / "loopqwen-assistance-study.pdf"
    assert (
        digest(previous)
        == "303a0fd3cc50564e565b35d66edb98bfb96537e7c231a14bae54f91a8c8babce"
    )
    from scripts.table_policy_answer_concentration import run as concentration

    from scripts.table_reasoning_reading import run as source_reading

    concentration()
    source_reading("report")
    reasoning = typst.compile(
        str(P / "table-reasoning-reading.typ"), root=str(P.resolve().parent), timestamp=0
    )
    (P / "table-reasoning-reading.pdf").write_bytes(reasoning)
    for title, raw in (
        ("Four-layer direct comparison", new),
        ("Masked source reading of causal and conversational reasoning", reasoning),
        (
            "Changed evidence and identical answers",
            (P / "table-answer-concentration.pdf").read_bytes(),
        ),
        (
            "Native execution-learning methods",
            (P / "table-policy-learning.pdf").read_bytes(),
        ),
        ("Recurrent supervised fit", (P / "reentry-fit.pdf").read_bytes()),
        ("Prior experiments and full project methodology", previous.read_bytes()),
    ):
        pdf = PdfReader(BytesIO(raw))
        assert all(len(page.extract_text()) > 200 for page in pdf.pages)
        writer.append(pdf, outline_item=title)
    dest = P / "loopqwen-four-layer-study.pdf"
    with dest.open("wb") as file:
        writer.write(file)
    print("Complete paper", len(PdfReader(dest).pages), digest(dest))


if __name__ == "__main__":
    build()
