"""Report intact human and independent software-release questions from the paired action trial."""

import csv
import json

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions


def make(paper, results, result, finish):
    name = "report-action-field-01"
    folder = results / name
    if not (folder / "checksums.json").exists():
        return
    decision = result(name)
    panel = read(read(folder / "registration.json")["material"])
    with (paper / "report-action-requests.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "arm",
                "domain",
                "source",
                "question_id",
                "request",
                "prompt_tokens",
                "question",
                "reference",
                "publisher_expression",
                "visible_answer_json",
                "status",
                "executions_json",
                "correct_final",
                "operational",
                "operand_intervention",
                "first_seconds",
                "wall_seconds",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            for row, run in zip(
                panel["rows"], unpack(folder / (side + ".json.gz")), strict=True
            ):
                for i, (question, q) in enumerate(
                    zip(row["questions"], run["requests"], strict=True)
                ):
                    tr = q["trajectory"]
                    reference = (
                        row["gold"]
                        if "gold" in row
                        else (row["answers"][i] if "answers" in row else "")
                    )
                    out.writerow(
                        [
                            side,
                            row["family"],
                            row.get("cluster", row["id"]),
                            row["id"],
                            i + 1,
                            len(tr["turns"][0]["prompt_ids"]),
                            question.split("\n\n")[-1],
                            reference,
                            row.get("expression") or "",
                            json.dumps(tr["answer"], ensure_ascii=False),
                            tr["status"],
                            json.dumps(executions(tr), ensure_ascii=False),
                            q["score"]["correct_answer"],
                            q["score"]["success"],
                            q["score"].get("bound_operands", ""),
                            q["first_s"],
                            q["wall_s"],
                            q["peak"],
                        ]
                    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75))
    for ax, (fam, n, title) in zip(
        axes,
        [
            ("report", 24, "Human financial reports"),
            ("distribution", 8, "Real package distributions"),
        ],
        strict=True,
    ):
        for side, shift, c, label in [
            ("candidate", -0.17, "#11675d", "Trained loop"),
            ("control", 0.17, "#9d5932", "Ordinary Base"),
        ]:
            ops = decision["operations"][side]
            vals = [
                ops[fam + " lookup"]["correct_answer"],
                ops[fam + " arithmetic"]["correct_answer"],
                ops[fam + " arithmetic"]["success"],
                decision["pairs"][fam]["loop" if side == "candidate" else "base"],
            ]
            bars = ax.bar(
                [i + shift for i in range(4)], vals, width=0.32, color=c, label=label
            )
            ax.bar_label(bars, fontsize=7, padding=1)
        ax.set_xticks(
            range(4),
            ["Lookup", "Arithmetic\nanswer", "Grounded\narithmetic", "Both\nrequests"],
            fontsize=7,
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, n * 1.15)
        ax.set_ylabel(f"Correct out of {n}", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    finish(fig, "report-actions")
