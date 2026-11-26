"""Plot per-request end-to-end speed and paired correctness on deployable tasks."""

import csv
import gzip
import json

import matplotlib.pyplot as plt

from loopqwen.io import digest, read


def make(paper, results, result, finish):
    # Each item contributes one complete session, including all real callbacks.
    studies = [
        ("BFCL numeric", "tool-dialogue-typed-01", None, "success"),
        ("WikiSQL, cold", "tool-dialogue-table-transfer-01", None, "success"),
        ("Package chains", "tool-dialogue-package-chain-01", None, "success"),
        ("WikiSQL, parser", "table-parser-01", None, "success"),
        ("HotpotQA", "bridge-reading-01", None, "content"),
        (
            "ASDiv",
            "tool-dialogue-program-reasoning-01",
            ("source", "asdiv"),
            "correct_answer",
        ),
        (
            "SVAMP",
            "tool-dialogue-program-reasoning-01",
            ("source", "svamp"),
            "correct_answer",
        ),
        ("ASDiv, planner", "operand-planner-01", ("source", "asdiv"), "correct_answer"),
        ("SVAMP, planner", "operand-planner-01", ("source", "svamp"), "correct_answer"),
        (
            "WikiSQL, trained",
            "tool-dialogue-sql-learning-01",
            ("provider", "wikisql"),
            "success",
        ),
    ]
    counts = []
    for label, name, selection, endpoint in studies:
        directory = results / name
        if not (directory / "checksums.json").exists():
            continue
        result(name)
        spec = read(directory / "registration.json")
        items = read(spec["material"])["rows"]
        selected = {
            w["id"]
            for w in items
            if selection is None or str(w[selection[0]]).casefold() == selection[1]
        }
        arms = []
        for arm in ("candidate", "control"):
            source = directory / (arm + ".json.gz")
            if digest(source) != read(directory / "checksums.json")[source.name]:
                raise ValueError(f"{name}/{source.name} failed verification")
            with gzip.open(source, "rt") as handle:
                arms.append([r for r in json.load(handle) if r["id"] in selected])
        a, b = arms
        assert a and [r["id"] for r in a] == [r["id"] for r in b]
        delta = [
            int(x["score"][endpoint]) - int(y["score"][endpoint])
            for x, y in zip(a, b, strict=True)
        ]
        times = [sum(r.get("trajectory", r)["seconds"] for r in side) for side in arms]
        n = len(a)
        counts.append(
            (
                label,
                name,
                endpoint,
                n,
                *[sum(r["score"][endpoint] for r in side) for side in arms],
                delta.count(1),
                delta.count(-1),
                *times,
            )
        )
    with (paper / "field-performance.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "study",
                "endpoint",
                "n",
                "candidate_correct",
                "base_correct",
                "paired_wins",
                "paired_losses",
                "candidate_seconds",
                "base_seconds",
            ]
        )
        w.writerows(counts)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.7, max(2.15, 0.36 * len(counts) + 0.65)), sharey=True
    )
    for i, (_, _, _, n, a, b, _, _, sa, sb) in enumerate(counts):
        left.plot([100 * b / n, 100 * a / n], [i, i], color="#b0b7b5", linewidth=2)
        left.scatter(100 * b / n, i + 0.05, c="#9D5932", marker="s", s=27, zorder=2)
        left.scatter(100 * a / n, i - 0.05, c="#11675D", s=32, zorder=2)
        right.barh(i, sb / sa, color="#55748a", height=0.45)
        right.text(sb / sa + 0.12, i, f"{sb / sa:.1f}×", va="center", fontsize=8)
    left.set_yticks(range(len(counts)), [r[0] for r in counts])
    left.invert_yaxis()
    left.set(xlim=(-3, 103), xlabel="Correct requests (%)")
    left.set_xticks([0, 25, 50, 75, 100])
    left.scatter([], [], c="#11675D", label="Recurrent", s=25)
    left.scatter([], [], c="#9D5932", marker="s", label="Base", s=25)
    left.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.24),
        ncol=2,
        fontsize=8,
    )
    right.axvline(1, linewidth=0.75, color="black")
    right.set_xlim(0, max(r[-1] / r[-2] for r in counts) * 1.19)
    right.set_xlabel("Speedup over Base (×)")
    finish(fig, "field-performance")
