"""Publish the changed-file comparison, generated answers, native cost and complete edition."""

import csv
import json
from math import comb
from pathlib import Path

import matplotlib

from scripts.assemble_evidence_paper import edition

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from loopqwen.research.native_function_study import executions
from scripts.file_evidence_audit import NAME, PANEL

P = Path("paper")
OUT = Path("research/results") / NAME


def discordant_interval(wins, losses):
    """Invert equal-tailed binomial tests for the loop win share among discordant articles."""
    n = wins + losses
    if n == 0:
        return None

    def upper(k):
        if k == n:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            cdf = sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
            if cdf > 0.025:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return 1 - upper(losses), upper(wins)


def uncertainty(rows, records, result):
    summary = []
    lines = [
        '#table(columns:(1.4fr,1fr,1fr,1fr,1fr,1.8fr),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f2f6f4")},[Source],[Both],[Loop only],[Base only],[Neither],[Loop share, 95% interval],'
    ]
    for family in ("file", "inline"):
        x = result["pairs"][family]
        articles = len({r["cluster"] for r in rows if r["family"] == family})
        both = x["candidate"] - x["wins"]
        assert both == x["base"] - x["losses"]
        neither = articles - both - x["wins"] - x["losses"]
        ci = discordant_interval(x["wins"], x["losses"])
        note = "no discordance" if ci is None else f"{ci[0]:.2f} to {ci[1]:.2f}"
        lines.append(
            f"[{family.capitalize()}],[{both}],[{x['wins']}],[{x['losses']}],[{neither}],[{note}],"
        )
        summary.append(
            dict(
                family=family,
                articles=articles,
                both=both,
                loop_only=x["wins"],
                base_only=x["losses"],
                neither=neither,
                conditional_interval=ci,
            )
        )
    lines.append(")")
    for arm, title in (("candidate", "Loop"), ("control", "Base")):
        rr = [q for record in records[arm] for q in record["requests"]]
        turns = [t for q in rr for t in q["trajectory"]["turns"]]
        nt = sum(len(t["token_ids"]) for t in turns)
        events = sum(len(executions(q["trajectory"])) for q in rr)
        completed = sum(q["score"]["completed"] for q in rr)
        lines.append(
            f"{title} produced {nt:,} generated tokens in {len(turns)} model turns, executed {events} function calls and completed {completed}/{len(rr)} requests.\n"
        )
    (P / "file-evidence-uncertainty.typ").write_text("\n".join(lines) + "\n")
    save(P / "file-evidence-discordance.json", summary)


def publish():
    rows = read(PANEL)["rows"]
    d = read(OUT / "decision.json")
    fit = read(OUT / "fit.json")
    blind = read("docs/measurements/file-evidence-answer-reading.json")
    exposure = read("docs/measurements/file-evidence-exposure.json")
    edits = {
        str(r["revision"]): r
        for r in read("docs/measurements/file-evidence-wikipedia-sources.json")[
            "articles"
        ]
    }
    notes = {
        r["case"]: r
        for r in read("research/diagnostics/file-evidence-source-reading.json")
    }
    sides = ("candidate", "control")
    records = {a: unpack(OUT / (a + ".json.gz")) for a in sides}
    details = []
    for i, w in enumerate(rows):
        article = edits[w["revision"]]
        row = {k: w[k] for k in ("id", "family", "page", "claim", "evidence", "gold")}
        row.update(
            diff_url=article["url"],
            prior_url=f"https://en.wikipedia.org/w/index.php?oldid={article['parent']}",
            revised_url=f"https://en.wikipedia.org/w/index.php?oldid={w['revision']}",
            source_reading=notes[w["cluster"]]["rating"],
            source_reason=notes[w["cluster"]]["reason"],
            opening_tokens=len(w["initial_ids"]),
        )
        for a in sides:
            result = records[a][i]
            assert w["id"] == result["id"]
            q = result["requests"][0]
            tr = q["trajectory"]
            events = executions(tr)
            fields = {
                "answer": json.dumps(tr["answer"], ensure_ascii=False),
                "status": tr["status"],
                "first_call": json.dumps(events[0]["call"], ensure_ascii=False)
                if events
                else "",
                "callbacks": len(events),
                "correct_lookup": q["score"]["observed"],
                "correct_conclusion": q["score"]["correct_answer"],
                "grounded": q["score"]["success"],
                "first_seconds": q["first_s"],
                "wall_seconds": q["wall_s"],
                "peak_bytes": q["peak"],
                "tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
            }
            row.update({a + "_" + k: v for k, v in fields.items()})
        details.append(row)
    with (P / "file-evidence-requests.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=details[0])
        writer.writeheader()
        writer.writerows(details)
    uncertainty(rows, records, d)
    m = d["groups"]
    a, b = m["candidate"], m["control"]
    f = d["pairs"]["file"]
    z = d["pairs"]["inline"]
    archived = Path("research/results")
    tool = read(archived / "tool-dialogue-counterfactual-01/decision.json")["metrics"]
    role = read(archived / "role-binding-01/decision.json")["groups"]
    compact = read(archived / "revision-evidence-01/decision.json")["scores"]
    document = read(archived / "document-frequency-reader-01/decision.json")["groups"]
    mixed = read(archived / "release-mixed-field-01/decision.json")["by_family"]["all"]
    matched = [
        read(archived / n / "decision.json")["scores"]
        for n in (
            "native-induction-anchored-matched-01",
            "native-induction-anchored-replication-01",
        )
    ]
    panels = [
        (
            "Changed files, complete grounded pairs /16",
            f["candidate"],
            f["base"],
            "untouched Base",
        ),
        (
            "Changed inline passages, compact reader /24 pairs",
            compact["candidate"]["pairs"],
            compact["base"]["pairs"],
            "untouched Base",
        ),
        (
            "Field table lookups, grounded answers /32",
            role["candidate"]["table"]["grounded"],
            role["control"]["table"]["grounded"],
            "untouched Base",
        ),
        (
            "Tool catalog, grounded tasks /132",
            tool["candidate"]["success"],
            tool["control"]["success"],
            "untouched Base",
        ),
        (
            "Field release, mixed requests /50",
            mixed["candidate"]["correct"],
            mixed["control"]["correct"],
            "untouched Base",
        ),
        (
            "Complete real documents, correct values /3",
            document["candidate"]["content"],
            document["control"]["content"],
            "untouched Base",
        ),
        (
            "Induction, correct across two panels /128",
            sum(x["recurrent"]["correct"] for x in matched),
            sum(x["dense"]["correct"] for x in matched),
            "equally trained K1",
        ),
    ]
    intro = [
        '#table(columns:(3.2fr,.72fr,.72fr,1.4fr),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f2f6f4")},[Workload and score],[Loop],[Control],[Comparator],'
    ]
    intro.extend(
        f"[{label}],[{v}],[{w}],[{control}]," for label, v, w, control in panels
    )
    (P / "file-evidence-synthesis.typ").write_text("\n".join(intro) + ")\n")
    (P / "file-evidence-abstract.typ").write_text(
        f"Across 16 actual changed files, the new loop returned two correct judgments grounded in real lookups on *{f['candidate']}/16* article pairs. Untouched Base did so on *{f['base']}/16*. The loop gained {f['wins']} complete file pairs and lost {f['losses']} (two-sided exact paired p = {f['p']:.4g}). On 12 independent inline articles, requiring both versions correct gave {z['candidate']}/12 versus {z['base']}/12. Across the 56 user requests, the systems took {a['all']['wall_s']:.2f} and {b['all']['wall_s']:.2f} seconds, respectively.\n"
    )
    counts = exposure["sources"]
    parity = fit["parity"]
    original = read("research/sessions/window-1795755565.json")
    previous_peak = max(r["peak_bytes"] for r in original["jobs"]) / 2**30
    (P / "file-evidence-fit.typ").write_text(
        f"The two native layers share {fit['trainable']:,} trained adapter parameters. The optimizer completed {fit['updates']} AdamW updates over {fit['targets']:,} output-token targets. Of those, {counts['file lookup']['target_tokens']:,} supervised opening lookups, {counts['file conclusion']['target_tokens']:,} supervised post-observation conclusions, {counts['inline conclusion']['target_tokens']:,} supervised inline judgments and {counts['WikiSQL rehearsal']['target_tokens']:,} came from WikiSQL rehearsal. Each six-example batch contained two opposite-source pairs and two other turns. The loss combines ordinary response-token cross entropy with two context-specific logit margins on the first differing response token. Preparing frozen prefix states took {fit['prepare_seconds']:.2f} s and native optimization took {fit['optimizer_seconds']:.2f} s. The original six-at-once run used {previous_peak:.3f} GiB, crossed its 6.50 GiB ceiling and ended before evaluating either model. This fit used physical groups of two, two, one and one with one optimizer update per logical batch. On the first sampled batch the MLX gradients differed by {parity['relative_gradient_l2']:.5f} in relative L2 norm and the losses by {parity['absolute_loss_error']:.5f}. The registered limits were 0.08 and 0.06, respectively.\n"
    )
    costs = [
        (
            "Correct file judgments regardless of lookup",
            a["file"]["answers"],
            b["file"]["answers"],
        ),
        (
            "Correct file judgments with exact lookup",
            a["file"]["grounded"],
            b["file"]["grounded"],
        ),
        ("Correct inline judgments", a["inline"]["answers"], b["inline"]["answers"]),
        ("Completed user requests", a["all"]["completed"], b["all"]["completed"]),
        ("Stream wall time (s)", a["all"]["wall_s"], b["all"]["wall_s"]),
        ("Summed first-token waits (s)", a["all"]["first_s"], b["all"]["first_s"]),
        (
            "Peak MLX allocation (GiB)",
            a["all"]["peak"] / 2**30,
            b["all"]["peak"] / 2**30,
        ),
        (
            "Opening input tokens / first-token seconds",
            a["opening_tps"],
            b["opening_tps"],
        ),
        ("Decode tokens / native decode seconds", a["decode_tps"], b["decode_tps"]),
    ]
    from statistics import median

    warm = {}
    for arm in sides:
        reqs = [r["requests"][0] for r in records[arm]]
        warm[arm] = {
            "first_request": reqs[0]["wall_s"],
            "peak_later": max(q["peak"] for q in reqs[1:]) / 2**30,
            "median": median(q["wall_s"] for q in reqs[1:]),
            "first_tokens": sum(
                len(q["trajectory"]["turns"][0]["prompt_ids"]) for q in reqs
            ),
            "native_first": sum(q["trajectory"]["turns"][0]["ttft_s"] for q in reqs),
        }
    costs.extend(
        [
            (
                "First cold request, seconds",
                warm["candidate"]["first_request"],
                warm["control"]["first_request"],
            ),
            (
                "Later requests, median seconds",
                warm["candidate"]["median"],
                warm["control"]["median"],
            ),
            (
                "Peak after first request, GiB",
                warm["candidate"]["peak_later"],
                warm["control"]["peak_later"],
            ),
            (
                "Native opening input tokens/s",
                warm["candidate"]["first_tokens"] / warm["candidate"]["native_first"],
                warm["control"]["first_tokens"] / warm["control"]["native_first"],
            ),
        ]
    )
    as_text = lambda n: f"{n:.2f}" if isinstance(n, float) else str(n)
    lines = [
        '#table(columns:(2.8fr,1fr,1fr),align:(left,right,right),stroke:none,inset:3.5pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f2f6f4")},[Measure],[Trained loop],[Untouched Base],'
    ]
    lines.extend(f"[{label}],[{as_text(v)}],[{as_text(w)}]," for label, v, w in costs)
    (P / "file-evidence-costs.typ").write_text("\n".join(lines) + ")\n")
    l, r = blind["candidate"], blind["control"]
    overlap = read("docs/measurements/file-evidence-source-overlap.json")["fields"]
    assert (
        not overlap["claim"]["normalized_exact"]
        and not overlap["claim"]["above_point_six"]
    )
    names = {
        "file_pairs": "changed-file accuracy",
        "inline_pairs": "inline-pair accuracy",
        "polarities": "accuracy on both labels in both settings",
        "completion": "completion",
        "wall_s": "wall time",
        "first_s": "first-token delay",
        "peak": "peak allocation",
        "opening_tps": "opening rate",
        "decode_tps": "decode rate",
    }
    missed = [v for k, v in names.items() if not d["gates"][k]]
    outcome = (
        "Every registered threshold passed."
        if d["passed"]
        else "The registered thresholds missed were "
        + ", ".join(missed)
        + ". The fit did not qualify for promotion, independent replication or longer context."
    )
    (P / "file-evidence-reading.typ").write_text(
        f"The loop obtained the intended source in {l['file']['correct_lookup']}/32 file requests, versus {r['file']['correct_lookup']}/32 for Base. Both versions of the right cell were returned on {l['file']['two_returned']} and {r['file']['two_returned']} article pairs, respectively. A blind reading of {blind['distinct_reviewed']} distinct free-form answers gives {l['file']['complete_pairs']} versus {r['file']['complete_pairs']} complete grounded file pairs, and {l['inline']['complete_pairs']} versus {r['inline']['complete_pairs']} complete inline pairs. On the {l['file']['direct_pairs']} file pairs with directly distinguishable publisher labels the corresponding counts were {l['file']['correct_direct_pairs']} and {r['file']['correct_direct_pairs']}. After also examining wording overlap, a separate exclusion of article pairs with five-word Jaccard similarity of at least 0.60 to a number-normalized training sentence leaves {l['file']['distinct_wording_pairs']} file and {l['inline']['distinct_wording_pairs']} inline pairs. The blinded paired scores there were {l['file']['correct_distinct_wording_pairs']} versus {r['file']['correct_distinct_wording_pairs']} on files and {l['inline']['correct_distinct_wording_pairs']} versus {r['inline']['correct_distinct_wording_pairs']} inline. {outcome}\n"
    )
    plt.rcParams.update(
        {
            "svg.hashsalt": "file-evidence-01",
            "font.size": 9,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 2.65), layout="constrained")
    labels = [
        "Right lookup\n/32",
        "Grounded answer\n/32",
        "Both files correct\n/16",
        "Both inline correct\n/12",
    ]
    for i, (arm, color, label) in enumerate(
        zip(sides, ("#17645b", "#8e4f2c"), ("Two-layer recurrent", "Untouched Base"))
    ):
        vals = [
            blind[arm]["file"]["correct_lookup"],
            m[arm]["file"]["grounded"],
            f["candidate" if i == 0 else "base"],
            z["candidate" if i == 0 else "base"],
        ]
        bars = ax.bar(
            [j + 0.36 * (i - 0.5) for j in range(4)],
            vals,
            width=0.34,
            color=color,
            label=label,
        )
        ax.bar_label(bars, padding=1)
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Count", ylim=(0, 37))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.17), frameon=False)
    fig.savefig(P / "file-evidence.svg", metadata={"Date": None})
    plt.close(fig)
    # One dot per real file request. The row labels refer to the actual changed article.
    from matplotlib.lines import Line2D

    files = [r for r in details if r["family"] == "file"]
    articles = sorted({r["page"] for r in files})
    labels = [
        s.replace("-COLON-", ":")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("2020 coronavirus pandemic in ", "2020 reports · ")
        for s in articles
    ]
    fig, grid = plt.subplots(figsize=(7.5, 4.55), layout="constrained")
    symbols = {
        (True, True): ("o", "#146953", "Correct after reading"),
        (True, False): ("s", "#d38b30", "Read, wrong answer"),
        (False, True): ("D", "#46669b", "Correct without the intended read"),
        (False, False): ("x", "#828184", "No grounded success"),
    }
    for y, article in enumerate(articles):
        for ai, arm in enumerate(sides):
            for gi, gold in enumerate(("yes", "no")):
                r = next(r for r in files if r["page"] == article and r["gold"] == gold)
                marker, color, _ = symbols[
                    r[arm + "_correct_lookup"], r[arm + "_correct_conclusion"]
                ]
                grid.scatter(ai * 2.7 + gi, y, marker=marker, s=49, c=color)
    grid.set(
        yticks=range(len(articles)),
        yticklabels=labels,
        xticks=[0, 1, 2.7, 3.7],
        xticklabels=["Loop · yes", "Loop · no", "Base · yes", "Base · no"],
        xlim=(-0.6, 4.3),
        ylim=(len(articles) - 0.35, -0.65),
    )
    grid.tick_params(axis="y", length=0, labelsize=8)
    grid.xaxis.tick_top()
    grid.spines[["right", "top", "left", "bottom"]].set_visible(False)
    grid.set_axisbelow(True)
    grid.grid(axis="y", alpha=0.17)
    handles = [
        Line2D([], [], color=c, marker=m, ls="", markersize=6, label=l)
        for m, c, l in symbols.values()
    ]
    fig.legend(
        handles=handles, loc="outside lower center", frameon=False, ncol=2, fontsize=8
    )
    fig.savefig(P / "file-evidence-articles.svg", metadata={"Date": None})
    plt.close(fig)
    # First article in the preregistered order, irrespective of either system's result.
    case = next(w["cluster"] for w in rows if w["family"] == "file")
    example = [v for w, v in zip(rows, details) if w["cluster"] == case]
    for row in example:
        for arm in sides:
            raw = json.loads(row[arm + "_answer"])
            row[arm + "_excerpt"] = (
                "No final answer."
                if not raw
                else (raw[:250] + "…" if len(raw) > 250 else raw)
            )
    save(P / "file-evidence-example.json", example)
    edition(P, "file-evidence")


if __name__ == "__main__":
    publish()
