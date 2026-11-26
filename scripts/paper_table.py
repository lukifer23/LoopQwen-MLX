"""Audit the first proposed SQL action and final executed outcome against the publisher."""

import csv
from decimal import Decimal, InvalidOperation

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.evaluation.table_questions import score
from loopqwen.io import digest, read
from loopqwen.table_spans import COMPARISONS, OPERATIONS


def filter_set(item, conditions):
    def literal(column, value):
        if item["types"][column] == "real":
            try:
                return str(Decimal(str(value).strip()).normalize())
            except InvalidOperation:
                pass
        return str(value).casefold()

    return sorted((c, op, literal(c, v)) for c, op, v in conditions)


def audit(item, trace):
    columns = item["columns"]
    gold = item["sql"]
    call = trace["turns"][0].get("call") or {}
    args = call.get("arguments") or {}
    fields = []
    valid = call.get("name") == "query_table" and isinstance(
        args.get("filters", []), list
    )
    if valid:
        for f in args.get("filters", []):
            if (
                not isinstance(f, dict)
                or f.get("column") not in columns
                or f.get("operator") not in COMPARISONS
                or "value" not in f
            ):
                valid = False
                break
            fields.append(
                (
                    columns.index(f["column"]),
                    COMPARISONS.index(f["operator"]),
                    f["value"],
                )
            )
    a = valid and args.get("operation") == OPERATIONS[gold["agg"]]
    c = valid and args.get("column") == columns[gold["sel"]]
    f = valid and filter_set(item, fields) == filter_set(item, gold["conds"])
    grade = score(item, trace)
    return a, c, f, (a and c and f), grade["path"], grade["success"]


def make(paper, results, result, finish):
    studies = [
        ("tool-dialogue-sql-learning-01", "SQL-trained", ("Recurrent", "Base")),
        ("table-parser-01", "Learned parser", ("Recurrent", "Base")),
        ("schema-calls-01", "Typed tool model", ("Guided", "Free")),
    ]
    fields = (
        "operation",
        "output_column",
        "filters",
        "whole_action",
        "executed_result",
        "answered_and_grounded",
    )
    summary = []
    with (paper / "sql-action-audit.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["study", "side", "question_id", "operation_family", *fields])
        for name, title, labels in studies:
            root = results / name
            if not (root / "checksums.json").exists():
                continue
            result(name)
            spec = read(root / "registration.json")
            checks = read(root / "checksums.json")
            items = {w["id"]: w for w in read(spec["material"])["rows"] if "sql" in w}
            for side, label in zip(("candidate", "control"), labels, strict=True):
                f = root / (side + ".json.gz")
                if digest(f) != checks[f.name]:
                    raise ValueError("Changed SQL trajectories")
                counts = [0] * len(fields)
                ids = []
                for r in unpack(f):
                    if r["id"] not in items:
                        continue
                    w = items[r["id"]]
                    observed = audit(w, r["trajectory"])
                    if observed[-1] != r["score"]["success"]:
                        raise ValueError("Execution outcome changed")
                    writer.writerow(
                        [
                            name,
                            side,
                            r["id"],
                            OPERATIONS[w["sql"]["agg"]],
                            *map(int, observed),
                        ]
                    )
                    counts = [x + y for x, y in zip(counts, observed, strict=True)]
                    ids.append(r["id"])
                if ids != list(items):
                    raise ValueError("Incomplete or reordered table trial")
                summary.append((title, side, label, len(ids), counts))
    if not summary:
        return
    fig, axes = plt.subplots(
        1, len(summary) // 2, figsize=(8.2, 2.85), sharey=True, squeeze=False
    )
    for panel, ax in enumerate(axes[0]):
        for title, side, label, n, counts in summary[panel * 2 : panel * 2 + 2]:
            dy, color, marker = (
                (-0.11, "#11675d", "o")
                if side == "candidate"
                else (0.11, "#9d5932", "s")
            )
            ax.scatter(
                [100 * x / n for x in counts],
                [i + dy for i in range(len(fields))],
                color=color,
                marker=marker,
                s=24,
                label=label,
            )
        ax.set(
            xlim=(-4, 104),
            xlabel="Correct (%)",
            title=f"{title}\n{n} tables",
        )
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0, 0].set_yticks(
        range(len(fields)),
        [
            "Operation",
            "Output column",
            "All filters",
            "Full first action",
            "Executed result",
            "Grounded answer",
        ],
    )
    axes[0, 0].invert_yaxis()
    finish(fig, "sql-action-audit")
