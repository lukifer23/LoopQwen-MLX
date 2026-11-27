"""Report quality and cold service time for real book questions followed by file operations."""

import csv
import json

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions


def make(paper, results, result, finish):
    name = "book-conversation-field-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    result(name)
    spec = read(root / "registration.json")
    items = read(spec["material"])["items"]
    fig, axes = plt.subplots(1, len(items), figsize=(7.1, 3.05), sharey=True)
    with (paper / "book-conversation-requests.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "book",
                "source_sha256",
                "arm",
                "request",
                "question",
                "reference",
                "answer_json",
                "status",
                "correct",
                "exact_call",
                "called_and_observed",
                "opening_tokens",
                "reused_on_open",
                "first_seconds",
                "wall_seconds",
                "peak_bytes",
                "kv_sites",
            ]
        )
        for i, (item, ax) in enumerate(zip(items, axes, strict=True)):
            for side, color, shift, label in [
                ("candidate", "#11675d", -0.16, "Released loop"),
                ("control", "#9d5932", 0.16, "Ordinary Base"),
            ]:
                rows = unpack(root / f"{i}-{side}.json.gz")
                x = [j + shift for j in range(len(rows))]
                vals = [r["wall_s"] for r in rows]
                ax.bar(x, vals, width=0.3, color=color, label=label)
                for j, r in enumerate(rows):
                    tr = r["trajectory"]
                    t = tr["turns"][0]
                    ax.annotate(
                        "yes" if r["score"]["success"] else "no",
                        (x[j], vals[j]),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
                    w.writerow(
                        [
                            item["title"],
                            item["sha256"],
                            side,
                            j + 1,
                            [
                                item["question"],
                                "Check actual file bytes",
                                "Check actual file lines",
                            ][j],
                            [item["reference"], item["size"], item["lines"]][j],
                            json.dumps(tr["answer"], ensure_ascii=False),
                            tr["status"],
                            r["score"]["success"],
                            r["score"]["exact_path"],
                            json.dumps(executions(tr), ensure_ascii=False),
                            len(t["prompt_ids"]),
                            t["reused_prompt_tokens"],
                            r["first_s"],
                            r["wall_s"],
                            r["peak"],
                            len(t["kv_history"]),
                        ]
                    )
            ax.set_title(item["title"], fontsize=8.5)
            ax.set_xticks(
                [0, 1, 2], ["Human question", "File bytes", "File lines"], fontsize=7.5
            )
            ax.set_yscale("log")
            ax.margins(y=0.18)
    axes[0].set_ylabel("Complete request seconds (log scale)")
    axes[-1].legend(frameon=False, fontsize=7)
    finish(fig, "book-conversation")
