"""Plot the cost of reading a new file and two subsequent live-tool questions."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions


def make(paper, results, result, finish):
    name = "conversation-document-prefill-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    result(name)
    spec = read(root / "registration.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), sharey=True)
    with (paper / "conversation-document-prefill.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "policy",
                "exchange",
                "initial_tokens",
                "reused_on_open",
                "answer",
                "path_grounded_and_correct",
                "calls",
                "first_seconds",
                "wall_seconds",
                "peak_bytes",
            ]
        )
        for i, (ax, item) in enumerate(zip(axes, spec["documents"], strict=True)):
            for side, delta, color, label in [
                ("stock", -0.17, "#879698", "Per request"),
                ("retain", 0.17, "#11675d", "Across messages"),
            ]:
                rows = unpack(root / f"{i}-{side}.json.gz")
                ax.bar(
                    [k + delta for k in range(len(rows))],
                    [q["first"] for q in rows],
                    width=0.32,
                    color=color,
                    label=label,
                )
                for j, q in enumerate(rows):
                    tr = q["trajectory"]
                    t = tr["turns"][0]
                    w.writerow(
                        [
                            item["source"],
                            side,
                            j + 1,
                            len(t["prompt_ids"]),
                            t.get("reused_prompt_tokens", 0),
                            tr["answer"],
                            int(q["score"]["success"]),
                            len(executions(tr)),
                            q["first"],
                            q["wall"],
                            q["peak"],
                        ]
                    )
            ax.set_title(["Project review", "Plotting source"][i], fontsize=9)
            ax.set_xticks(
                [0, 1, 2], ["Read file", "Check bytes", "Check lines"], fontsize=8
            )
    axes[0].set_ylabel("Seconds to first token")
    axes[1].legend(ncol=1, frameon=False, fontsize=8)
    finish(fig, "conversation-document-prefill")
