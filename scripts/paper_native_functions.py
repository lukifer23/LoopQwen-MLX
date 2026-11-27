"""Export both user exchanges, callback evidence and paired outcomes for official Qwen tools."""

import csv
import json

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions


def make(paper, results, result, finish):
    name = "native-function-dialogue-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    panel = read(read(root / "registration.json")["material"])["rows"]
    write_requests(paper / "native-function-requests.csv", root, panel)
    groups = ["ASDiv", "SVAMP", "public", "packages", "files"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.95), sharey=True)
    for ax, key, denom, title in zip(
        axes,
        ("strict", "answers"),
        ("dialogues", "requests"),
        ("Complete grounded dialogues", "Correct answers per request"),
        strict=True,
    ):
        for side, shift, color, label in [
            ("candidate", -0.15, "#11675d", "Two passes"),
            ("control", 0.15, "#9d5932", "Base"),
        ]:
            bank = decision["groups"][side]
            vals = [100 * bank[g][key] / bank[g][denom] for g in groups]
            bars = ax.barh(
                [i + shift for i in range(len(groups))],
                vals,
                height=0.28,
                color=color,
                label=label,
            )
            ax.bar_label(
                bars,
                labels=[f"{bank[g][key]}/{bank[g][denom]}" for g in groups],
                padding=2,
                fontsize=7,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 131)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Percent")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(
        range(len(groups)),
        ["ASDiv", "SVAMP", "BFCL", "Package versions", "Local files"],
    )
    axes[0].invert_yaxis()
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(-0.05, -0.29),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    finish(fig, "native-function-dialogues")


def write_requests(path, root, panel, *, json_answers=False):
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "family",
                "arm",
                "exchange",
                "question",
                "status",
                "answer_json" if json_answers else "answer",
                "correct_answer",
                "strict_success",
                "call_and_observation",
                "first_seconds",
                "wall_seconds",
                "peak_mlx_bytes",
                "output_tokens",
            ]
        )
        for side in ("candidate", "control"):
            for w, r in zip(panel, unpack(root / (side + ".json.gz")), strict=True):
                for i, (question, q) in enumerate(
                    zip(w["questions"], r["requests"], strict=True), 1
                ):
                    tr = q["trajectory"]
                    writer.writerow(
                        [
                            r["id"],
                            r["family"],
                            side,
                            i,
                            question,
                            tr["status"],
                            json.dumps(tr["answer"], ensure_ascii=False)
                            if json_answers
                            else tr["answer"],
                            q["score"]["correct_answer"],
                            q["score"]["success"],
                            json.dumps(executions(tr), ensure_ascii=False),
                            q["first_s"],
                            q["wall_s"],
                            q["peak"],
                            sum(len(t["token_ids"]) for t in tr["turns"]),
                        ]
                    )
