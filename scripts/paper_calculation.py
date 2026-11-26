"""Report numerical answers and real callback evidence on all held-out documents."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from paper_overlap import audit

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.tool_session import compact


def make(paper, results, result, finish):
    name = "report-calculation-01"
    spec = read(Path("research/experiments") / (name + ".json"))
    mat = read(spec["material"])
    panel = mat["rows"]
    train = read(spec["training"])
    fitting = [w["question"] for w in train["episodes"]]
    typed_ids = {
        w["id"]
        for w in read(Path("research/data/tool-dialogue-typed-training.json"))[
            "examples"
        ]
    }
    replay = [w for w in train["examples"] if w["id"] in typed_ids]
    assert len(replay) == 192
    fitting.extend(
        m["content"] for w in replay for m in w["messages"] if m["role"] == "user"
    )
    with (paper / "report-calculation-inputs.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "source",
                "numbers_masked",
                "fit_questions",
                "test_questions",
                "exact",
                "near",
                "max_fivegram_jaccard",
                "clusters",
                "min_prompt",
                "max_prompt",
            ]
        )
        for source in ("FinQA", "TabMWP"):
            cohort = [w for w in panel if w["source"] == source]
            lengths = [len(w["initial_ids"]) for w in cohort]
            for mask in (False, True):
                out.writerow(
                    [
                        source,
                        mask,
                        *audit(fitting, [w["question"] for w in cohort], mask),
                        len({w["cluster"] for w in cohort}),
                        min(lengths),
                        max(lengths),
                    ]
                )
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    d = result(name)
    sums = read(root / "checksums.json")
    with (paper / "report-calculation-requests.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "item",
                "source",
                "issuer_or_table",
                "side",
                "question",
                "gold",
                "tolerance",
                "answer",
                "correct",
                "strict",
                "grounded",
                "linked",
                "status",
                "events",
                "prompt_tokens",
                "generated_tokens",
                "first_s",
                "total_s",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for w, r in zip(panel, unpack(path), strict=True):
                assert w["id"] == r["id"]
                tr, s = r["trajectory"], r["score"]
                turns = tr["turns"]
                events = [
                    {"call": t.get("call"), "observation": t.get("observation")}
                    for t in turns
                    if "call" in t
                ]
                out.writerow(
                    [
                        w["id"],
                        w["source"],
                        w["cluster"],
                        side,
                        w["question"],
                        w["gold"],
                        w["tolerance"],
                        tr["answer"],
                        s["correct"],
                        s["strict"],
                        s["grounded"],
                        s["linked_grounded"],
                        tr["status"],
                        compact(events),
                        len(w["initial_ids"]),
                        sum(len(t["token_ids"]) for t in turns),
                        turns[0]["ttft_s"],
                        tr["seconds"],
                        max(t["peak_bytes"] for t in turns),
                    ]
                )
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.3), sharey=True)
    for ax, key, label in zip(
        axes,
        ("correct", "grounded", "linked_grounded"),
        ("Final number", "Calculator-grounded", "Prior return reused"),
        strict=True,
    ):
        for i, source in enumerate(("FinQA", "TabMWP")):
            for side, shift, color, marker, title in (
                ("candidate", -0.12, "#11675d", "o", "Recurrent"),
                ("control", 0.12, "#9d5932", "s", "Base"),
            ):
                y = d["groups"][source][side][key]
                ax.scatter(
                    y,
                    i + shift,
                    color=color,
                    marker=marker,
                    label=title if i == 0 else None,
                )
        ax.set_xlim(-1, 25)
        ax.set_xticks([0, 8, 16, 24])
        ax.set_xlabel(label)
    axes[0].set_yticks([0, 1], ["FinQA", "TabMWP"])
    axes[0].set_ylim(1.45, -0.45)
    axes[0].legend(
        frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=2, fontsize=8
    )
    finish(fig, "report-calculation")
