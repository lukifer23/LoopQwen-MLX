"""Summarize every human arithmetic request and the program actually executed."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.tool_session import compact


def make(paper, results, result, finish):
    name = "draft-program-01"
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    outcome = result(name)
    sums = read(root / "checksums.json")
    subjects = read(read(root / "registration.json")["material"])["rows"]
    fields = ("success", "path", "correct_answer", "strict_answer", "completed")
    with (paper / "draft-program-requests.csv").open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "id",
                "source",
                "model",
                "question",
                "gold",
                "final_answer",
                "calls_and_observations",
                "draft_length",
                "proposed_boundary",
                "output_tokens",
                "wall_s",
                "first_token_s",
                *fields,
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for r, item in zip(unpack(path), subjects, strict=True):
                assert r["id"] == item["id"]
                tr = r["trajectory"]
                turns = tr["turns"]
                obs = [
                    {k: t[k] for k in ("call", "observation") if k in t}
                    for t in turns
                    if "call" in t
                ]
                rec = r["draft"]
                w.writerow(
                    [
                        r["id"],
                        item["source"],
                        side,
                        item["question"],
                        item["gold"],
                        tr["answer"],
                        compact(obs),
                        len(rec["ids"]) if rec else 0,
                        rec["proposed_boundary"] if rec else None,
                        sum(len(t["token_ids"]) for t in turns),
                        tr["seconds"],
                        turns[0]["ttft_s"],
                        *[r["score"][k] for k in fields],
                    ]
                )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    groups = outcome["groups"]
    labels = ["all", *sorted(k for k in groups if k != "all")]
    for i, key in enumerate(labels):
        for side, offset, color, marker, label in (
            ("candidate", -0.10, "#11675d", "o", "Two-pass finisher"),
            ("control", 0.10, "#9d5932", "s", "Base"),
        ):
            cell = groups[key][side]
            for ax, metric in zip(
                axes, ("correct_answer", "path", "success"), strict=True
            ):
                ax.scatter(
                    100 * cell[metric] / cell["n"],
                    i + offset,
                    marker=marker,
                    color=color,
                    s=30,
                    label=label if i == 0 else None,
                )
    axes[0].set_yticks(
        range(len(labels)), ["All 48", *[s + " (24)" for s in labels[1:]]]
    )
    axes[0].invert_yaxis()
    for ax, title in zip(
        axes,
        (
            "Correct final quantity",
            "Correct callback result",
            "Both in a completed reply",
        ),
        strict=True,
    ):
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(title + " (%)", fontsize=8)
    axes[0].legend(
        loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=2, frameon=False, fontsize=8
    )
    finish(fig, "draft-program")
