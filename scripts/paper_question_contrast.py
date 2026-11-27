"""Plot completed goals on the same passage and free numerical reasoning by source."""

import csv

import matplotlib.pyplot as plt
from paper_native_functions import write_requests

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read


def overlap(paper, spec):
    from paper_overlap import audit
    from question_contrast_material import READING

    from loopqwen.evaluation.program_questions import INSTRUCTION

    train = read(spec["training"])["examples"]
    all_users = {
        m["content"] for x in train for m in x["messages"] if m["role"] == "user"
    }

    def parts(text):
        source = text.removesuffix(READING).removesuffix(INSTRUCTION)
        if text.endswith(READING):
            passage, question = source.rsplit("\n\n", 1)
            return question, passage
        return source, None

    train_questions, train_passages = zip(*(parts(s) for s in all_users))
    held = read(spec["material"])["rows"]
    panels = [
        (
            "DROP passages",
            [v for v in train_passages if v],
            [parts(w["questions"][0])[1] for w in held if w["family"] == "DROP"],
        ),
    ]
    for family in ("DROP", "ASDiv", "public", "files", "packages"):
        panels.append(
            (
                family + " user questions",
                train_questions,
                [
                    parts(q)[0]
                    for w in held
                    if w["family"] == family
                    for q in w["questions"]
                ],
            )
        )
    with (paper / "question-contrast-overlap.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "items",
                "numbers_masked",
                "unique_training",
                "unique_test",
                "exact",
                "nonexact_jaccard_at_least_0.6",
                "maximum_five_word_jaccard",
            ]
        )
        for label, fit, test in panels:
            for mask in (False, True):
                out.writerow([label, mask, *audit(fit, test, mask)])


def make(paper, results, result, finish):
    name = "question-contrast-01"
    root = results / name
    summary = result(name)
    spec = read(root / "registration.json")
    overlap(paper, spec)
    rows = read(spec["material"])["rows"]
    write_requests(
        paper / "question-contrast-requests.csv", root, rows, json_answers=True
    )
    fig, ax = plt.subplots(figsize=(7, 2.65))
    labels = [
        "DROP questions",
        "Both from a passage",
        "ASDiv final number",
        "ASDiv grounded",
        "Public calls",
        "Package pairs",
        "File pairs",
    ]
    with (paper / "question-contrast-passages.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "passage",
                "arm",
                "question1",
                "reference1",
                "answer1",
                "question2",
                "reference2",
                "answer2",
                "both_correct",
            ]
        )
        for side, delta, color, label in [
            ("candidate", -0.16, "#11675d", "Loop"),
            ("control", 0.16, "#9d5932", "Base"),
        ]:
            passages = {}
            for w, r in zip(rows, unpack(root / (side + ".json.gz")), strict=True):
                if w["family"] == "DROP":
                    passages.setdefault(w["cluster"], []).append((w, r["requests"][0]))
            for pid, items in passages.items():
                values = []
                for w, q in items:
                    values.extend(
                        [
                            w["questions"][0].split("\n\n")[-1].split("\n")[0],
                            w["gold"],
                            q["trajectory"]["answer"],
                        ]
                    )
                writer.writerow(
                    [pid, side, *values, all(q["score"]["success"] for _, q in items)]
                )
            d = summary["groups"][side]
            values = [
                (d["DROP"]["answers"], 48),
                (summary["whole_passages"][side], 24),
                (d["ASDiv"]["answers"], 24),
                (d["ASDiv"]["grounded"], 24),
                (d["public"]["strict"], 22),
                (d["packages"]["strict"], 6),
                (d["files"]["strict"], 6),
            ]
            bars = ax.bar(
                [i + delta for i in range(len(labels))],
                [100 * v / n for v, n in values],
                width=0.31,
                color=color,
                label=label,
            )
            ax.bar_label(
                bars, labels=[str(v) for v, _ in values], fontsize=7, padding=1
            )
    ax.set_xticks(
        range(len(labels)), [v.replace(" ", "\n", 1) for v in labels], fontsize=7.5
    )
    ax.set(ylim=(0, 118), ylabel="Correct (%)")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "question-contrast")
