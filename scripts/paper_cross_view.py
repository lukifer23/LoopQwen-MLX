"""Publish measured cross-view reasoning, model costs, original answers and consolidated paper."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import NAME, tokenizer
from loopqwen.research.native_function_study import executions
from scripts.assemble_evidence_paper import edition
from scripts.cross_view_uncertainty import paired_articles

P = Path("paper")


def build():
    root = Path("research/results") / NAME
    s = read(root / "registration.json")
    d = read(root / "decision.json")
    fit = read(root / "fit.json")
    ts = read(root / "teacher-summary.json")
    study = read(s["material"])
    items = study["rows"]
    groups = d["groups"]
    a, b = groups["candidate"], groups["control"]
    exposure = read("docs/measurements/cross-view-exposure.json")
    blind = read("docs/measurements/cross-view-answer-reading.json")
    runs = {k: unpack(root / (k + ".json.gz")) for k in groups}
    close_reasoning = tokenizer(s).convert_tokens_to_ids("</think>")
    records = []
    for i, w in enumerate(items):
        out = {
            k: w.get(k)
            for k in (
                "id",
                "question_id",
                "cluster",
                "family",
                "gold",
                "other",
                "source",
                "document",
            )
        }
        out["question"] = w["questions"][0]
        out["expected_calls"] = json.dumps(w["calls"], ensure_ascii=False)
        for arm, rr in runs.items():
            r = rr[i]
            q = r["requests"][0]
            tr = q["trajectory"]
            sequences = [t["token_ids"] for t in tr["turns"]]
            assert r["id"] == w["id"]
            record = {
                **q["score"],
                "answer": tr["answer"],
                "status": tr["status"],
                "events": json.dumps(executions(tr), ensure_ascii=False),
                "wall_s": q["wall_s"],
                "first_s": q["first_s"],
                "peak": q["peak"],
                "output_tokens": sum(map(len, sequences)),
                "reasoning_prefix_tokens": sum(
                    ids.index(close_reasoning) if close_reasoning in ids else len(ids)
                    for ids in sequences
                ),
                "model_turns": len(sequences),
                "turns_without_reasoning_close": sum(
                    close_reasoning not in ids for ids in sequences
                ),
            }
            out.update({arm + "_" + k: v for k, v in record.items()})
        records.append(out)
    with (P / "cross-view-requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    questions = [q for a in study["articles"] for q in a["questions"]]
    modes = {
        (r["question_id"], r["family"]): r
        for r in records
        if r["family"] != "retention"
    }
    plt.rcParams.update(
        {
            "svg.hashsalt": "cross-view-reading-01",
            "font.size": 9,
            "font.family": "DejaVu Sans",
        }
    )
    colors = ["#17645b", "#619caa", "#cf8943", "#d9dbdc"]
    fig, ax = plt.subplots(figsize=(7.45, 2.05), layout="constrained")
    labels = ["Both views", "Inline only", "File only", "Neither"]
    channel = {}
    for i, (arm, name) in enumerate(
        (("candidate", "Trained loop"), ("control", "Untouched Base"))
    ):
        counts = [0] * 4
        for q in questions:
            iv = modes[q["id"], "inline"][arm + "_success"]
            fv = modes[q["id"], "file"][arm + "_success"]
            counts[
                {
                    (True, True): 0,
                    (True, False): 1,
                    (False, True): 2,
                    (False, False): 3,
                }[iv, fv]
            ] += 1
        channel[arm] = dict(zip(labels, counts, strict=True))
        left = 0
        for j, n in enumerate(counts):
            ax.barh(
                i,
                n,
                left=left,
                color=colors[j],
                height=0.53,
                label=labels[j] if i == 0 else None,
            )
            if n:
                ax.text(
                    left + n / 2,
                    i,
                    str(n),
                    ha="center",
                    va="center",
                    color="white" if j == 0 else "#161b1d",
                )
            left += n
        assert left == len(questions)
    ax.set(
        yticks=[0, 1],
        yticklabels=["Trained loop", "Untouched Base"],
        xlim=(0, len(questions)),
        xlabel="Original human questions",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.savefig(P / "cross-view-channels.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.45, 4.50), layout="constrained")
    style = {
        "correct": ("o", "#17645b"),
        "read_wrong": ("s", "#d38b30"),
        "miss": ("x", "#8b8b89"),
    }
    arts = sorted(study["articles"], key=lambda a: a["page"])
    for j, art in enumerate(arts):
        for side, arm in enumerate(("candidate", "control")):
            for mi, mod in enumerate(("inline", "file")):
                for qi, q in enumerate(art["questions"]):
                    r = modes[q["id"], mod]
                    code = (
                        "correct"
                        if r[arm + "_success"]
                        else "read_wrong"
                        if mod == "file" and r[arm + "_observed"]
                        else "miss"
                    )
                    marker, color = style[code]
                    ax.scatter(
                        5 * side + 2 * mi + qi, j, marker=marker, color=color, s=23
                    )
    ax.set(
        yticks=range(len(arts)),
        yticklabels=[a["page"] for a in arts],
        xticks=[0.5, 2.5, 5.5, 7.5],
        xticklabels=["Loop inline", "Loop file", "Base inline", "Base file"],
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", alpha=0.13)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], marker=style[k][0], color=style[k][1], ls="", label=v)
            for k, v in (
                ("correct", "Successful"),
                ("read_wrong", "Right read, wrong reply"),
                ("miss", "Other miss"),
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(P / "cross-view-articles.svg", metadata={"Date": None})
    plt.close(fig)
    lines = [
        '#table(columns:(2.8fr,1fr,1fr),stroke:none,inset:3.4pt,align:(left,right,right),fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f2f6f4")},[Measure],[Trained loop],[Untouched Base],'
    ]
    for f in ("file", "inline"):
        for k, label in (
            ("grounded", "successful"),
            ("exact", "exact short answers"),
            ("mean_f1", "mean token F1"),
        ):
            x, y = a[f][k], b[f][k]
            lines.append(
                f"[{f.capitalize()} {label}],[{x:.3f}],[{y:.3f}],"
                if k == "mean_f1"
                else f"[{f.capitalize()} {label} /40],[{x}],[{y}],"
            )
    for lab, x, y in (
        ("Right file return /40", a["file"]["right_reads"], b["file"]["right_reads"]),
        (
            "Grounded table retention /8",
            a["retention"]["grounded"],
            b["retention"]["grounded"],
        ),
        ("Completed /88", a["all"]["completed"], b["all"]["completed"]),
        ("Entire stream, seconds", a["all"]["wall_s"], b["all"]["wall_s"]),
        ("First-token wait sum, seconds", a["all"]["first_s"], b["all"]["first_s"]),
        ("Opening tokens/s", a["opening_tps"], b["opening_tps"]),
        ("Decode tokens/s", a["decode_tps"], b["decode_tps"]),
        ("Peak MLX GiB", a["all"]["peak"] / 2**30, b["all"]["peak"] / 2**30),
    ):
        fmt = lambda z: f"{z:.2f}" if isinstance(z, float) else str(z)
        lines.append(f"[{lab}],[{fmt(x)}],[{fmt(y)}],")
    (P / "cross-view-metrics.typ").write_text("\n".join(lines) + ")\n")
    first = study["articles"][0]
    msg = []
    for q in first["questions"]:
        f = modes[q["id"], "file"]
        inline = modes[q["id"], "inline"]

        def short(k, record):
            reply = record[k + "_answer"]
            return (
                (reply[:105] + "…" if len(reply) > 105 else reply)
                if reply
                else "no completed answer"
            )

        msg.append(
            f"For “{q['question']}” the publisher answer is “{q['answer']}”. The loop answered “{short('candidate', inline)}” inline and “{short('candidate', f)}” with the file. Base answered “{short('control', inline)}” and “{short('control', f)}”."
        )
    missed = [k for k, v in d["gates"].items() if not v]
    verdict = (
        "All registered thresholds passed."
        if d["passed"]
        else "The candidate missed its registered qualification. The failed thresholds were "
        + ", ".join(k.replace("_", " ") for k in missed)
        + "."
    )
    checks = blind["arms"]
    generation = []
    for arm, name in (("candidate", "The trained loop"), ("control", "Base")):

        def total(field):
            return sum(r[arm + "_" + field] for r in records)

        assert total("output_tokens") == groups[arm]["all"]["output_tokens"]
        generation.append(
            f"{name} generated {total('output_tokens'):,} tokens in {total('model_turns')} model turns. It emitted {total('reasoning_prefix_tokens'):,} tokens before the first reasoning-close marker, including the entire generated prefix on {total('turns_without_reasoning_close')} turns that never emitted that marker."
        )
    uncertainty = paired_articles(records)
    save(P / "cross-view-uncertainty.json", uncertainty)
    primary = uncertainty["paired"]["all_four"]
    assert (primary["wins"], primary["losses"], primary["paired_p"]) == (
        d["wins"],
        d["losses"],
        d["p"],
    )
    intervals = [
        '#table(columns:(2.25fr,1fr,1.7fr),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f2f6f4")},[Paired measure],[Loop − Base],[95% interval],'
    ]
    for key, label in (
        ("all_four", "All four answers"),
        ("file", "File success"),
        ("inline", "Inline success"),
        ("file_f1", "File token F1"),
        ("inline_f1", "Inline token F1"),
    ):
        v = uncertainty["paired"][key]
        lo, hi = v["interval"]
        intervals.append(
            f"[{label}],[{v['difference']:+.3f}],[{lo:+.3f} to {hi:+.3f}],"
        )
    (P / "cross-view-intervals.typ").write_text("\n".join(intervals) + ")\n")
    repeats = []
    for arm, name in (("candidate", "The loop"), ("control", "Base")):
        items = uncertainty["two_completed"][arm]

        def count_repeated(f):
            return f"{sum(x['repeated_answer'] for x in items[f])}/{len(items[f])}"

        repeats.append(
            f"{name} gave the same normalized reply to both different questions on {count_repeated('file')} file articles and {count_repeated('inline')} inline articles with two completed replies."
        )
    activation = read("docs/measurements/cross-view-activation.json")
    assert activation["compiled_weights"] == read(root / "compiled.json")["weights"]
    value = {
        "generation": " ".join(generation),
        "repetitions": " ".join(repeats),
        "intervention": f"The physical probe uses the actual compiled weights and calls the project measurement file for the sum of first-token waits. The callback returned {activation['actual_observation']['value']} seconds. Both final-layer visits independently retained all {len(activation['prompt_ids'])} tokens of the resulting conversation. Setting the repeated residual contribution to zero changed the next-token logits by RMS {activation['rms_logit_change']:.4f}. Restoring the coefficient reproduced all {activation['vocabulary_size']:,} logits bit for bit.",
        "channels": channel,
        "summary": f"The trained loop completed all four judgments on {a['both_modes_both_questions']}/20 articles and untouched Base on {b['both_modes_both_questions']}/20. The loop gained {d['wins']} articles and lost {d['losses']} (two-sided exact paired p = {d['p']:.4g}). The 88 requests took {a['all']['wall_s']:.2f} and {b['all']['wall_s']:.2f} seconds respectively.",
        "decision": verdict,
        "training": f"The teacher covered {ts['first_gold_top_k']}/{ts['examples']} first human answer tokens in its top 32, with {exposure['first_answers_top_k']['1']} ranked first. The registered minimum top-32 coverage was 65%. Teacher targets took {ts['seconds']:.2f} seconds and {ts['peak_bytes'] / 2**30:.2f} GiB peak. The student updated {fit['trainable']:,} parameters over {fit['updates']} AdamW steps and {fit['targets']:,} total target-token exposures across {exposure['unique_examples']:,} distinct student examples. It received {fit['teacher_positions']:,} supervised answer-position distributions. The frozen-prefix and optimization stages took {fit['prepare_seconds']:.2f} and {fit['optimizer_seconds']:.2f} seconds. Its fit peaked at {fit['peak_bytes'] / 2**30:.2f} GiB. The seven projections were fused once into a {read(root / 'compiled.json')['bytes']:,}-byte layer. The registered analysis includes every user request.",
        "sensitivity": f"A single arm-blind project-agent reading covered {blind['distinct_reviewed']} distinct completed nonexact replies. Under that secondary interpretation the whole-article count was {checks['candidate']['all_four']}/20 versus {checks['control']['all_four']}/20. The exact and token-F1 results above preserve the original machine-scored references.",
        "example": f"The first enrolled article was {first['page']}. " + " ".join(msg),
    }
    save(P / "cross-view-paper.json", value)
    edition(P, "cross-view")


if __name__ == "__main__":
    build()
