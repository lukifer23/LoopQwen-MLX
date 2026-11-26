"""Paired evidence on human passage questions, grouped by original article."""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.evaluation.relations import cluster_randomization
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    name = "passage-copy-01"
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    totals = result(name)["groups"]
    items = read(read(root / "registration.json")["material"])["rows"]
    inventory = read(root / "checksums.json")
    arms = []
    for side in ("candidate", "control"):
        f = root / (side + ".json.gz")
        assert inventory[f.name] == digest(f)
        arms.append(unpack(f))

    differences = defaultdict(lambda: defaultdict(int))
    with (paper / "passage-answers.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "source",
                "title",
                "question",
                "human_references",
                "arm",
                "answer",
                "exact",
                "content",
                "f1",
                "completed",
                "generated_tokens",
                "seconds",
                "first_token_seconds",
                "peak_bytes",
            ]
        )
        for w, candidate, base in zip(items, *arms, strict=True):
            assert w["id"] == candidate["id"] == base["id"]
            delta = int(candidate["score"]["exact"]) - int(base["score"]["exact"])
            for subset in ["all", w["domain"]] + (
                ["unanswerable"] if w["unanswerable"] else []
            ):
                differences[subset][(w["domain"], w["title"])] += delta
            for side, row in (("candidate", candidate), ("control", base)):
                score = row["score"]
                writer.writerow(
                    [
                        w["id"],
                        w["domain"],
                        w["title"],
                        w["question"],
                        " | ".join(w["references"]),
                        side,
                        score["visible"],
                        int(score["exact"]),
                        int(score["content"]),
                        score["f1"],
                        int(score["completed"]),
                        len(row["ids"]),
                        row["seconds"],
                        row["ttft"],
                        row["peak"],
                    ]
                )

    fig, (exact, f1) = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    names = ["all", "SQuAD", "QuAC", "unanswerable"]
    with (paper / "passage-scores.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "subset",
                "articles",
                "items",
                "arm",
                "exact",
                "content",
                "f1",
                "completed",
                "seconds",
                "first_token_seconds",
                "peak_bytes",
                "paired_gains",
                "paired_losses",
                "item_exact_p",
                "article_randomization_p",
            ]
        )
        for i, group in enumerate(names):
            record = totals[group]
            by_title = differences[group]
            assert sum(by_title.values()) == record["gains"] - record["losses"]
            p = cluster_randomization(list(by_title.values()))
            for side, offset, color, marker, label in (
                ("candidate", -0.11, "#11675d", "o", "Two passes"),
                ("control", 0.11, "#9d5932", "s", "Base"),
            ):
                m = record[side]
                writer.writerow(
                    [
                        group,
                        len(by_title),
                        m["n"],
                        side,
                        m["exact"],
                        m["content"],
                        m["f1"],
                        m["completed"],
                        m["seconds"],
                        m["ttft"],
                        m["peak"],
                        record["gains"],
                        record["losses"],
                        record["p_exact"],
                        p,
                    ]
                )
                opts = {
                    "color": color,
                    "marker": marker,
                    "s": 30,
                    "label": label if i == 0 else None,
                }
                exact.scatter(100 * m["exact"] / m["n"], i + offset, **opts)
                f1.scatter(100 * m["f1"], i + offset, **opts)
    exact.set_yticks(
        range(4), ["All 64", "SQuAD (32)", "QuAC (32)", "QuAC unanswerable (8)"]
    )
    exact.invert_yaxis()
    for ax in (exact, f1):
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
    exact.set_xlabel("Completed exact match (%)")
    f1.set_xlabel("Mean token F1 (%)")
    exact.legend(loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=2, frameon=False)
    finish(fig, "passage-answers")
