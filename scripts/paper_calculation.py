"""Report numerical answers and real callback evidence on all held-out documents."""

import ast
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from paper_overlap import audit

from loopqwen.calculator_tool import execute
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.tool_session import compact


def operation(expression):
    node = ast.parse(expression.strip(), mode="eval").body
    if not isinstance(node, ast.BinOp):
        return "", []
    operands = [str(execute([ast.unparse(arg)])[0]) for arg in (node.left, node.right)]
    return type(node.op).__name__, operands


def first_steps(paper, panel, root, sums):
    with (paper / "report-first-calculations.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "id",
                "side",
                "expert_first_step",
                "first_executed_step",
                "expert_operator",
                "executed_operator",
                "expert_arguments",
                "executed_arguments",
                "same_operator",
                "arguments_in_common",
                "same_operation_and_ordered_values",
                "correct_final_answer",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for item, row in zip(panel, unpack(path), strict=True):
                if item["source"] != "FinQA":
                    continue
                assert item["id"] == row["id"]
                expr = next(
                    (
                        t["call"]["arguments"]["steps"][0]
                        for t in row["trajectory"]["turns"]
                        if "value" in t.get("observation", {})
                    ),
                    "",
                )
                expert = item["steps"][0]
                op, args = operation(expert)
                actual, vals = operation(expr) if expr else ("", [])
                assert op
                overlap = sum((Counter(args) & Counter(vals)).values())
                out.writerow(
                    [
                        item["id"],
                        side,
                        expert,
                        expr,
                        op,
                        actual,
                        compact(args),
                        compact(vals),
                        op == actual,
                        overlap,
                        op == actual and args == vals,
                        row["score"]["correct"],
                    ]
                )


def make(paper, results, result, finish):
    name = "report-calculation-01"
    spec = read(Path("research/experiments") / (name + ".json"))
    mat = read(spec["material"])
    panel = mat["rows"]
    train = read(spec["training"])
    fitting = [w["question"] for w in train["episodes"]]
    typed_ids = {
        w["id"]
        for w in read(Path("research/data/tool-dialogue-typed-training.json"))[
            "examples"
        ]
    }
    replay = [w for w in train["examples"] if w["id"] in typed_ids]
    assert len(replay) == 192
    fitting.extend(
        m["content"] for w in replay for m in w["messages"] if m["role"] == "user"
    )
    with (paper / "report-calculation-inputs.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "source",
                "numbers_masked",
                "fit_questions",
                "test_questions",
                "exact",
                "near",
                "max_fivegram_jaccard",
                "clusters",
                "min_prompt",
                "max_prompt",
            ]
        )
        for source in ("FinQA", "TabMWP"):
            cohort = [w for w in panel if w["source"] == source]
            lengths = [len(w["initial_ids"]) for w in cohort]
            for mask in (False, True):
                out.writerow(
                    [
                        source,
                        mask,
                        *audit(fitting, [w["question"] for w in cohort], mask),
                        len({w["cluster"] for w in cohort}),
                        min(lengths),
                        max(lengths),
                    ]
                )
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    d = result(name)
    sums = read(root / "checksums.json")
    first_steps(paper, panel, root, sums)
    with (paper / "report-calculation-requests.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "item",
                "source",
                "issuer_or_table",
                "side",
                "question",
                "gold",
                "tolerance",
                "answer",
                "correct",
                "strict",
                "grounded",
                "linked",
                "status",
                "events",
                "prompt_tokens",
                "generated_tokens",
                "first_s",
                "total_s",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for w, r in zip(panel, unpack(path), strict=True):
                assert w["id"] == r["id"]
                tr, s = r["trajectory"], r["score"]
                turns = tr["turns"]
                events = [
                    {"call": t.get("call"), "observation": t.get("observation")}
                    for t in turns
                    if "call" in t
                ]
                out.writerow(
                    [
                        w["id"],
                        w["source"],
                        w["cluster"],
                        side,
                        w["question"],
                        w["gold"],
                        w["tolerance"],
                        tr["answer"],
                        s["correct"],
                        s["strict"],
                        s["grounded"],
                        s["linked_grounded"],
                        tr["status"],
                        compact(events),
                        len(w["initial_ids"]),
                        sum(len(t["token_ids"]) for t in turns),
                        turns[0]["ttft_s"],
                        tr["seconds"],
                        max(t["peak_bytes"] for t in turns),
                    ]
                )
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.3), sharey=True)
    for ax, key, label in zip(
        axes,
        ("correct", "grounded", "linked_grounded"),
        ("Final number", "Calculator-grounded", "Prior return reused"),
        strict=True,
    ):
        for i, source in enumerate(("FinQA", "TabMWP")):
            for side, shift, color, marker, title in (
                ("candidate", -0.12, "#11675d", "o", "Recurrent"),
                ("control", 0.12, "#9d5932", "s", "Base"),
            ):
                y = d["groups"][source][side][key]
                ax.scatter(
                    y,
                    i + shift,
                    color=color,
                    marker=marker,
                    label=title if i == 0 else None,
                )
        ax.set_xlim(-1, 25)
        ax.set_xticks([0, 8, 16, 24])
        ax.set_xlabel(label)
    axes[0].set_yticks([0, 1], ["FinQA", "TabMWP"])
    axes[0].set_ylim(1.45, -0.45)
    axes[0].legend(
        frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=2, fontsize=8
    )
    finish(fig, "report-calculation")
