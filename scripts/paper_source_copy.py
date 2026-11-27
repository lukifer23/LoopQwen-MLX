"""Report the native query, training supervision and complete-document learning outcomes."""

from pathlib import Path
import matplotlib.pyplot as plt

from loopqwen.io import read, digest
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_reading import committed

P = Path("paper")
D = Path("docs/measurements")
R = Path("research/results")


def record(path):
    committed(path)
    return read(path)


def collect():
    attention = record(D / "bounded-read-source-address/summary.json")
    teacher = record(D / "conversational-training-teacher/receipt.json")
    data = record(Path("research/sources/source-copy/inventory.json"))
    review = record(Path("research/sources/source-copy/teacher-review.json"))
    panel = record(D / "source-copy-panel.json")
    rows = []
    for label, m in attention["sums"].items():
        group, depth = label.split("/")
        rows.append(
            dict(
                group=group,
                reads=int(depth),
                positions=int(m["n"]),
                **{
                    k: m[k] / m["n"]
                    for k in (
                        "source_hit",
                        "span_mass",
                        "copy_hit",
                        "native_hit",
                        "uniform_top",
                        "better_than_uniform",
                    )
                },
            )
        )
    write_csv(P / "source-address-probabilities.csv", rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.1), layout="constrained", sharey=True
    )
    labels = [
        "Peak in cited span",
        "Copied token correct",
        "Vocabulary token correct",
        "Frequency baseline",
    ]
    for ax, group, title in zip(
        axes,
        ("calculate", "reasoning"),
        ("Financial operand positions", "Reasoning evidence positions"),
        strict=True,
    ):
        colors = ("#92b2bb", "#2e6872")
        for j, r in enumerate(q for q in rows if q["group"] == group):
            ax.barh(
                [i + (j - 0.5) * 0.32 for i in range(4)],
                [
                    100 * r[k]
                    for k in ("source_hit", "copy_hit", "native_hit", "uniform_top")
                ],
                height=0.31,
                color=colors[j],
                label=f"{r['reads']} read" + ("s" if r["reads"] > 1 else ""),
            )
        ax.set(xlabel="Fraction of labeled decisions (%)", title=title, xlim=(0, 75))
        ax.set_yticks(range(4), labels)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(P / "source-address-probabilities.svg", metadata={"Date": None})
    plt.close(fig)
    out = dict(
        attention=attention,
        teacher=teacher,
        accepted=sum(r["include"] for r in review["review"]),
        material=data,
        panel=panel,
        registration=digest(Path("research/experiments/source-copy-fit-01.json")),
    )
    long = D / "changed-locale-fit/receipt.json"
    if long.exists():
        run = record(long)
        out["locale"] = {
            k: v for k, v in run.items() if k not in ("features", "history")
        }
        out["locale"]["updates"] = len(run["history"])
        if "held" in run:
            stages = ("initial", "one", "two")
            held = run["held"]
            curves = [
                dict(path=stage, **row) for stage in stages for row in held[stage]
            ]
            write_csv(P / "changed-locale-held.csv", curves)
            out["locale"]["dual"] = {
                stage: sum(
                    all(r["margin"] > 0 for r in held[stage] if r["key"] == k)
                    for k in {r["key"] for r in held[stage]}
                )
                for stage in stages
            }
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 3), layout="constrained")
            for ax, metric, label in zip(
                axes,
                ("margin", "address"),
                ("Correct minus changed-token logit", "Evidence loss (nats)"),
                strict=True,
            ):
                for i, base in enumerate(held["initial"]):
                    ax.plot(
                        range(3),
                        [held[k][i][metric] for k in stages],
                        color=("#337b83", "#bd7043")[base["version"]],
                        marker="o",
                        lw=1.1,
                        ms=3,
                        alpha=0.8,
                    )
                ax.set_xticks(range(3), ["Parent", "One read", "Two reads"])
                ax.set_ylabel(label)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].axhline(0, color="#777", lw=0.7)
            fig.savefig(P / "changed-locale-held.svg", metadata={"Date": None})
            plt.close(fig)
        if run["history"]:
            write_csv(P / "changed-locale-updates.csv", run["history"])
    for stage, key in (
        ("source-copy-fit-01", "fit"),
        ("source-copy-transfer-01", "pair"),
    ):
        if (R / stage / "checksums.json").is_file():
            record(R / stage / "checksums.json")
            out[key] = read(R / stage / "decision.json")
    assessed = D / "source-copy-reading/report.json"
    delivery = D / "source-copy-delivery.json"
    if assessed.is_file() and delivery.is_file():
        import csv

        reading = record(assessed)
        out["reading"] = reading
        out["delivery"] = record(delivery)
        with (P / "source-copy-requests.csv").open() as f:
            samples = list(csv.DictReader(f))
        labels = ("first", "live", "matched")
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.5), layout="constrained")
        for col, family in enumerate(("reading", "report")):
            ax, wall = axes[0, col], axes[1, col]
            for j, (arm, color, title) in enumerate(
                (("candidate", "#34737c", "Loop"), ("control", "#b46b44", "Base"))
            ):
                counts = []
                times = []
                for stage in labels:
                    cases = [
                        r
                        for r in samples
                        if r["family"] == family and r["presentation"] == stage
                    ]
                    counts.append(
                        reading["positions"][f"{family}/{stage}/{arm}"]["correct"]
                    )
                    times.append(sum(float(r[arm + "_wall_s"]) for r in cases))
                positions = [k + (j - 0.5) * 0.37 for k in range(3)]
                ax.bar(positions, counts, width=0.36, color=color, label=title)
                wall.bar(positions, times, width=0.36, color=color)
            ax.set_title(
                "Reading conversations"
                if family == "reading"
                else "Financial conversations"
            )
            ax.set_ylim(0, 24)
            ax.set_ylabel("Correct delivered conclusions / 24")
            wall.set_ylabel("Total wall time (s)")
            for panel in (ax, wall):
                panel.set_xticks(range(3), ["First", "Own history", "Fixed history"])
                panel.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False)
        fig.savefig(P / "source-copy-conversation-results.svg", metadata={"Date": None})
        plt.close(fig)
    audit = D / "source-copy-program-audit.json"
    if audit.exists():
        checked = record(audit)
        out["operative"] = {
            "results": checked["results"],
            "controls": sum(r["eligible"] for r in checked["controls"]),
            "perturbations": sum(r["tests"] for r in checked["controls"]),
        }
        stages = (
            ("At least one call", "executed"),
            ("At least one valid return", "valid"),
            ("Last valid return correct", "last_correct_return"),
            ("Tracks source operands", "last_operative"),
            ("Also delivered correctly", "last_operative_and_delivered"),
        )
        fig, ax = plt.subplots(figsize=(8, 2.9), layout="constrained")
        for offset, (arm, title, color) in enumerate(
            (
                ("candidate", "Two reads", "#34737c"),
                ("control", "Untouched Base", "#b46b44"),
            )
        ):
            positions = [j + 0.34 * (offset - 0.5) for j in range(len(stages))]
            values = [checked["results"][arm][k] for _, k in stages]
            bars = ax.barh(positions, values, height=0.34, color=color, label=title)
            ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_yticks(range(len(stages)), [title for title, _ in stages])
        ax.invert_yaxis()
        ax.set_xlim(0, 72)
        ax.set_xlabel("Financial question presentations (72 per model)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.savefig(P / "source-copy-program-audit.svg", metadata={"Date": None})
        plt.close(fig)
    return out
