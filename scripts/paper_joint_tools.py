"""Publish matched gains, losses and time on the four preregistered tool panels."""

import csv
from collections import Counter

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    def records(name, side):
        result(name)
        path = results / name / (side + ".json.gz")
        if digest(path) != read(results / name / "checksums.json")[path.name]:
            raise ValueError("Different tool transcript")
        return {r["id"]: r for r in unpack(path)}

    name = "tool-dialogue-counterfactual-01"
    if not (results / name / "checksums.json").exists():
        return
    old = records("tool-dialogue-typed-01", "candidate") | records(
        "tool-dialogue-unified-01", "candidate"
    )
    new = records(name, "candidate")
    base = records(name, "control")
    items = read(read(results / name / "registration.json")["material"])["rows"]
    common = {
        w["id"]: w
        for w in read("research/data/tool-dialogue-unified-01.json")
        + read("research/data/tool-exec-bfcl-01.json")["rows"]
    }
    for w in items:
        if w["id"] in common:
            assert all(w[k] == v for k, v in common[w["id"]].items())
            assert (
                new[w["id"]]["trajectory"]["turns"][0]["prompt_ids"]
                == old[w["id"]]["trajectory"]["turns"][0]["prompt_ids"]
            )
    sources = [
        ("Original catalogs", "catalog"),
        ("Grounded catalogs", "grounded"),
        ("Public numerical", "public"),
        ("New domains", "fresh"),
    ]
    rows = []
    for label, panel in sources:
        ids = [w["id"] for w in items if w["panel"] == panel]

        def compare(arm):
            relevant = [k for k in ids if k in arm]
            deltas = Counter(
                int(new[k]["score"]["success"]) - int(arm[k]["score"]["success"])
                for k in relevant
            )
            return (
                (
                    sum(arm[k]["score"]["success"] for k in relevant),
                    deltas[1],
                    deltas[-1],
                )
                if relevant
                else ("", "", "")
            )

        rows.append(
            [
                label,
                len(ids),
                sum(new[k]["score"]["success"] for k in ids),
                *compare(base),
                *compare(old),
                sum(new[k]["trajectory"]["seconds"] for k in ids),
                sum(base[k]["trajectory"]["seconds"] for k in ids),
            ]
        )
    with (paper / "joint-tools.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "panel",
                "n",
                "joint_correct",
                "base_correct",
                "wins_over_base",
                "losses_to_base",
                "prior_correct",
                "wins_over_prior",
                "losses_to_prior",
                "joint_seconds",
                "base_seconds",
            ]
        )
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 2.45))
    for i, (label, n, a, b, _, _, prior, _, _, _, _) in enumerate(rows):
        ax.plot([100 * b / n, 100 * a / n], [i, i], color="#cbd0cf", lw=2)
        ax.scatter(100 * b / n, i, c="#9d5932", marker="s", s=36)
        ax.scatter(100 * a / n, i - 0.055, c="#11675d", marker="o", s=40)
        if prior != "":
            ax.scatter(100 * prior / n, i + 0.055, c="#536a89", marker="^", s=43)
    ax.scatter([], [], c="#11675d", marker="o", label="New recurrent")
    ax.scatter([], [], c="#536a89", marker="^", label="Earlier recurrent")
    ax.scatter([], [], c="#9d5932", marker="s", label="Base")
    ax.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(xlim=(-2, 102), xlabel="Correct complete tool sessions (%)")
    finish(fig, "joint-tools")
