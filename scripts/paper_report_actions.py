"""Report intact human and independent software-release questions from the paired action trial."""

import csv
import json

import matplotlib.pyplot as plt
from paper_overlap import audit, normalize
from report_action_material import INSTRUCTION, source
from report_units_audit import visible_scale

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions
from loopqwen.research.observation_review_study import paired


def overlap(paper, training, panel):
    users = {
        m["content"]
        for r in training["examples"]
        for m in r["messages"]
        if m["role"] == "user"
    }
    texts = [r for r in users if r.endswith(INSTRUCTION)]
    contexts = [r.removesuffix(INSTRUCTION).rsplit("\n\n", 1)[0] for r in texts]
    questions = [r.removesuffix(INSTRUCTION).rsplit("\n\n", 1)[-1] for r in users]
    with (paper / "report-action-overlap.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "comparison",
                "mask_numbers",
                "distinct_train",
                "distinct_held",
                "exact",
                "nonexact_jaccard_at_least_0.6",
                "max_five_word_jaccard",
            ]
        )
        for domain in ("report", "distribution"):
            held = [
                w["questions"][0].removesuffix(INSTRUCTION).rsplit("\n\n", 1)
                for w in panel["rows"]
                if w["family"] == domain
            ]
            for label, fit, idx in (
                ("questions", questions, 1),
                ("source documents", contexts, 0),
            ):
                for mask in (False, True):
                    out.writerow(
                        [
                            domain + " " + label,
                            mask,
                            *audit(fit, [t[idx] for t in held], mask),
                        ]
                    )


def disjoint_report_pairs(paper, training, panel, folder):
    known = {
        normalize(m["content"].removesuffix(INSTRUCTION).rsplit("\n\n", 1)[-1], True)
        for r in training["examples"]
        for m in r["messages"]
        if m["role"] == "user"
    }
    excluded = {
        w["cluster"]
        for w in panel["rows"]
        if w["family"] == "report"
        and normalize(
            w["questions"][0].removesuffix(INSTRUCTION).rsplit("\n\n", 1)[1], True
        )
        in known
    }
    good = {}
    for side in ("candidate", "control"):
        clusters = {}
        arm = unpack(folder / (side + ".json.gz"))
        for w, r in zip(panel["rows"], arm, strict=True):
            if w["family"] == "report" and w["cluster"] not in excluded:
                clusters.setdefault(w["cluster"], []).append(
                    r["requests"][0]["score"]["success"]
                )
        assert all(len(x) == 2 for x in clusters.values())
        good[side] = {k for k, v in clusters.items() if all(v)}
    p = paired(good["candidate"], good["control"])
    with (paper / "report-action-disjoint.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "excluded_whole_reports",
                "retained_whole_reports",
                "loop_pairs",
                "base_pairs",
                "loop_gains",
                "loop_losses",
                "paired_p",
            ]
        )
        out.writerow(
            [
                " ".join(sorted(excluded)),
                len(clusters),
                len(good["candidate"]),
                len(good["control"]),
                p["wins"],
                p["losses"],
                p["p"],
            ]
        )


def make(paper, results, result, finish):
    name = "report-action-field-01"
    folder = results / name
    if not (folder / "checksums.json").exists():
        return
    decision = result(name)
    spec = read(folder / "registration.json")
    panel = read(spec["material"])
    training = read(spec["training"])
    overlap(paper, training, panel)
    disjoint_report_pairs(paper, training, panel, folder)
    with (paper / "report-action-requests.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "arm",
                "domain",
                "source",
                "question_id",
                "request",
                "prompt_tokens",
                "question",
                "reference",
                "publisher_expression",
                "visible_answer_json",
                "status",
                "executions_json",
                "correct_final",
                "operational",
                "operand_intervention",
                "first_seconds",
                "wall_seconds",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            for row, run in zip(
                panel["rows"], unpack(folder / (side + ".json.gz")), strict=True
            ):
                for i, (question, q) in enumerate(
                    zip(row["questions"], run["requests"], strict=True)
                ):
                    tr = q["trajectory"]
                    reference = (
                        row["gold"]
                        if "gold" in row
                        else (row["answers"][i] if "answers" in row else "")
                    )
                    out.writerow(
                        [
                            side,
                            row["family"],
                            row.get("cluster", row["id"]),
                            row["id"],
                            i + 1,
                            len(tr["turns"][0]["prompt_ids"]),
                            question.split("\n\n")[-1],
                            reference,
                            row.get("expression") or "",
                            json.dumps(tr["answer"], ensure_ascii=False),
                            tr["status"],
                            json.dumps(executions(tr), ensure_ascii=False),
                            q["score"]["correct_answer"],
                            q["score"]["success"],
                            q["score"].get("bound_operands", ""),
                            q["first_s"],
                            q["wall_s"],
                            q["peak"],
                        ]
                    )
    publisher = {q["uid"]: q for d in source("dev") for q in d["questions"]}
    with (paper / "report-action-units.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "arm",
                "question_id",
                "question",
                "reference",
                "reference_scale",
                "visible_answer_json",
                "status",
                "registered_numeric",
                "scale_interpretation",
                "scale_credited",
            ]
        )
        for side in ("candidate", "control"):
            for w, r in zip(
                panel["rows"], unpack(folder / (side + ".json.gz")), strict=True
            ):
                if w["family"] != "report":
                    continue
                q = r["requests"][0]
                tr = q["trajectory"]
                human = publisher[w["id"]]
                correct, interpretation = visible_scale(
                    tr["answer"], human["scale"], w["gold"]
                )
                out.writerow(
                    [
                        side,
                        w["id"],
                        human["question"],
                        w["gold"],
                        human["scale"],
                        json.dumps(tr["answer"], ensure_ascii=False),
                        tr["status"],
                        q["score"]["correct_answer"],
                        interpretation,
                        correct and tr["status"] == "answered",
                    ]
                )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75))
    for ax, (fam, n, title) in zip(
        axes,
        [
            ("report", 24, "Human financial reports"),
            ("distribution", 8, "Real package distributions"),
        ],
        strict=True,
    ):
        for side, shift, c, label in [
            ("candidate", -0.17, "#11675d", "Trained loop"),
            ("control", 0.17, "#9d5932", "Ordinary Base"),
        ]:
            ops = decision["operations"][side]
            vals = [
                ops[fam + " lookup"]["correct_answer"],
                ops[fam + " arithmetic"]["correct_answer"],
                ops[fam + " arithmetic"]["success"],
                decision["pairs"][fam]["loop" if side == "candidate" else "base"],
            ]
            bars = ax.bar(
                [i + shift for i in range(4)], vals, width=0.32, color=c, label=label
            )
            ax.bar_label(bars, fontsize=7, padding=1)
        ax.set_xticks(
            range(4),
            ["Lookup", "Arithmetic\nanswer", "Grounded\narithmetic", "Both\nrequests"],
            fontsize=7,
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, n * 1.15)
        ax.set_ylabel(f"Correct out of {n}", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    finish(fig, "report-actions")
