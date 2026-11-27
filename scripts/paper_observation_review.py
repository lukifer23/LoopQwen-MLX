"""Show free response accuracy alongside individual accept/retry margins."""

import csv

import matplotlib.pyplot as plt
from paper_native_functions import write_requests
from review_first_actions import write as write_actions

from loopqwen.io import read


def make(paper, results, result, finish):
    name = "observation-review-01"
    root = results / name
    decision = result(name)
    panel = read(read(root / "registration.json")["material"])
    write_requests(
        paper / "observation-review-requests.csv",
        root,
        panel["rows"],
        json_answers=True,
    )
    write_actions(root, paper / "observation-review-first-actions.csv")
    fig, (accuracy, margins) = plt.subplots(1, 2, figsize=(7.15, 3.2))
    colors = {"candidate": "#11675d", "control": "#9d5932"}
    with (paper / "observation-review-decisions.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "question",
                "arm",
                "right_expression",
                "wrong_expression",
                "right_margin",
                "wrong_margin",
                "both_decisions_correct",
            ]
        )
        for side, delta, label in [
            ("candidate", -0.18, "Loop review"),
            ("control", 0.18, "Base"),
        ]:
            groups = decision["groups"][side]
            values = [
                100 * groups[f][key] / groups[f][n]
                for f, key, n in [
                    ("ASDiv", "answers", "requests"),
                    ("SVAMP", "answers", "requests"),
                    ("ASDiv", "grounded", "requests"),
                    ("SVAMP", "grounded", "requests"),
                    ("public", "strict", "dialogues"),
                    ("packages", "strict", "dialogues"),
                    ("files", "strict", "dialogues"),
                ]
            ]
            accuracy.barh(
                [i + delta for i in range(7)],
                values,
                height=0.3,
                color=colors[side],
                label=label,
            )
            points = []
            for w, r in zip(
                panel["probe"], read(root / (side + "-probe.json")), strict=True
            ):
                assert w["id"] == r["id"]
                yes, no = (
                    r[k]["logits"][0] - r[k]["logits"][1] for k in ("right", "wrong")
                )
                writer.writerow(
                    [
                        r["id"],
                        w["question"],
                        side,
                        w["right"]["arguments"]["steps"],
                        w["wrong"]["arguments"]["steps"],
                        yes,
                        no,
                        yes > 0 and no < 0,
                    ]
                )
                points.append((no, yes))
            xs, ys = zip(*points, strict=True)
            margins.scatter(
                xs,
                ys,
                c=colors[side],
                s=18,
                alpha=0.7,
                marker="o" if side == "candidate" else "x",
            )
    accuracy.set_yticks(
        range(7),
        [
            "ASDiv final",
            "SVAMP final",
            "ASDiv grounded",
            "SVAMP grounded",
            "Public calls",
            "Package pairs",
            "File pairs",
        ],
    )
    accuracy.invert_yaxis()
    accuracy.set(xlim=(0, 100), xlabel="Success (%)", title="Free exchanges")
    accuracy.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    margins.axhline(0, lw=0.7, color="#454545")
    margins.axvline(0, lw=0.7, color="#454545")
    margins.set(
        xlabel="Answer − retry on wrong return",
        ylabel="Answer − retry on correct return",
        title="Paired logit margins",
    )
    for ax in (accuracy, margins):
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "observation-review")
