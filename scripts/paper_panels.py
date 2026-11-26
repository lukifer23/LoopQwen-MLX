"""Plot independent human-reading decisions and single-checkpoint tool transfer."""

import csv

import matplotlib.pyplot as plt

READING = {
    "contrast-boolq-recurrent-01": "Base-initialized reading fit",
    "contrast-boolq-transfer-01": "Paired inference transfer",
}
TOOLS = {
    "tool-dialogue-typed-01": ("Published numerical executions", "Base", "success"),
    "tool-dialogue-typed-depth-01": ("Same executions", "trained K1", "success"),
    "tool-dialogue-instruct-01": ("Same executions", "untouched Instruct", "success"),
    "tool-bfcl-multiple-01": ("Multiple available schemas", "Base", "correct"),
    "tool-bfcl-recovery-01": ("Earlier single-schema panel", "Base", "correct"),
    "tool-dialogue-unified-01": ("Catalog and multi-turn retention", "Base", "success"),
    "tool-dialogue-package-chain-01": (
        "Live package dependency chains",
        "Base",
        "success",
    ),
    "tool-dialogue-table-transfer-01": (
        "Queries on human-authored tables",
        "Base",
        "success",
    ),
}


def generate(paper, results, result, finish):
    colors = ["#11675d", "#9d5932"]
    completed = [
        (name, label)
        for name, label in READING.items()
        if (results / name / "checksums.json").is_file()
    ]
    if completed:
        fig, axes = plt.subplots(
            len(completed), 1, figsize=(6.7, 1.8 * len(completed) + 0.5), squeeze=False
        )
        for ax, (name, label) in zip(axes[:, 0], completed, strict=True):
            r = result(name)
            y = range(3)
            ax.set_title(label, fontsize=10)
            for offset, side, color in zip(
                [-0.16, 0.16], ["candidate", "control"], colors, strict=True
            ):
                vals = [r["metrics"][side][k] for k in ("pairs", "correct", "probe")]
                ns = [r["pairs"], r["items"], r["items"]]
                bars = ax.barh(
                    [i + offset for i in y],
                    [100 * v / n for v, n in zip(vals, ns, strict=True)],
                    height=0.26,
                    color=color,
                    label="Recurrent" if side == "candidate" else "Base",
                )
                ax.bar_label(
                    bars,
                    labels=[f"{v}/{n}" for v, n in zip(vals, ns, strict=True)],
                    padding=3,
                    fontsize=8,
                )
            ax.set_yticks(
                y,
                [
                    "Both questions correct",
                    "Individual answers",
                    "Shared-prefix decisions",
                ],
            )
            ax.invert_yaxis()
            ax.set_xlim(0, 112)
            ax.set_xticks([0, 25, 50, 75, 100])
        axes[0, 0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.43),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        axes[-1, 0].set_xlabel("Correct (%)")
        finish(fig, "human-reading")
    rows = []
    for name, (label, control, endpoint) in TOOLS.items():
        if not (results / name / "checksums.json").is_file():
            continue
        r = result(name)
        c, b = [r["metrics"][k] for k in ("candidate", "control")]
        rows.append(
            [
                name,
                label,
                control,
                endpoint,
                r["items"],
                c[endpoint],
                b[endpoint],
                c["seconds"],
                b["seconds"],
                r["gains"],
                r["losses"],
                r["p_exact"],
                r["passed"],
            ]
        )
    with (paper / "typed-transfer.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "study",
                "workload",
                "comparator",
                "endpoint",
                "items",
                "recurrent",
                "control",
                "recurrent_seconds",
                "control_seconds",
                "paired_wins",
                "paired_losses",
                "p_exact",
                "gate_passed",
            ]
        )
        w.writerows(rows)
    if len(rows) > 1:
        fig, ax = plt.subplots(figsize=(7.2, 0.43 * len(rows) + 1.45))
        for i, r in enumerate(rows):
            _, _, _, _, n, a, b, *_ = r
            ax.plot([100 * a / n, 100 * b / n], [i, i], zorder=1, color="#98aaa5", lw=2)
            ax.scatter(100 * a / n, i - 0.04, color=colors[0], s=36, zorder=3)
            ax.scatter(
                100 * b / n, i + 0.04, color=colors[1], marker="s", s=35, zorder=3
            )
            ax.text(104, i, f"{a} / {b} of {n}", va="center", fontsize=8)
        ax.set_yticks(range(len(rows)), [r[1] + " · " + r[2] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(-3, 136)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Successful requests (%)")
        for label, color, mark in zip(
            ["Frozen typed recurrent model", "Comparator"], colors, ["o", "s"]
        ):
            ax.scatter([], [], c=color, marker=mark, label=label)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.46, 1.23),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        finish(fig, "typed-transfer")
