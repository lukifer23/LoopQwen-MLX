"""Summarize the frozen training intervention and its independent source checks."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from scripts.assistance_deliberation import write_csv
from scripts.learner_prefix_material import RECEIPT, RECORDS
from scripts.register_learner_prefix import FIT, PRIMARY, RESULT, TARGETS
from scripts.table_policy_reading import committed

P = Path("paper")
DOC = Path("docs/measurements")


def collect():
    inputs = (RECORDS, RECEIPT, TARGETS, DOC / "learner-prefix-panel.json")
    for p in inputs:
        committed(p)
    rows = unpack(TARGETS)
    counts = Counter(w["region"] for w in rows)
    spec = read(Path("research/experiments") / (FIT + ".json"))
    report = {
        "training": read(RECEIPT),
        "native_tokens": sum(len(w["tokens"]) for w in unpack(RECORDS)),
        "regions": counts,
        "epochs": spec["epochs"],
        "positions": spec["expected_positions"],
        "probes": len(spec["probes"]),
        "panel": read(inputs[-1]),
        "lr": spec["lr"],
        "kl": spec["teacher_kl"],
    }
    root = RESULT / FIT
    if (root / "checksums.json").is_file():
        committed(root / "checksums.json")
        report["fit"] = read(root / "decision.json")
        jobs = [
            job
            for window in Path("research/sessions").glob("window-*.json")
            for job in read(window)["jobs"]
            if job["job"] == FIT
        ]
        if jobs:
            report["fit_cost"] = {
                "seconds": sum(job["seconds"] for job in jobs),
                "peak": max(job["peak_bytes"] for job in jobs),
            }
        fix = root / "frozen-teacher-correction.json"
        if fix.is_file():
            report["teacher_fix"] = read(fix)
        replay = DOC / (FIT + "-cold.json")
        if replay.is_file():
            committed(replay)
            report["replay"] = read(replay)
        if (root / "training-log.json.gz").is_file():
            history = unpack(root / "training-log.json.gz")
            records = [
                {
                    "update": i,
                    **{
                        k: r[k]
                        for k in (
                            "epoch",
                            "nll",
                            "gradient",
                            "loss",
                            "parent_kl",
                            "positions",
                        )
                    },
                }
                for i, r in enumerate(history, 1)
            ]
            write_csv(P / "learner-prefix-updates.csv", records)
            g = report["fit"]["gains"]
            labels = (
                "strategy",
                "report_action",
                "report_reply",
                "repair_known_schema",
                "redirect_valid_read",
                "answer_valid_read",
            )
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), layout="constrained")
            for i, (kind, color) in enumerate(
                (("clean", "#34666c"), ("learner", "#ad6846"))
            ):
                data = [
                    (j, g[k + "/" + kind])
                    for j, k in enumerate(labels)
                    if k + "/" + kind in g
                ]
                axes[0].barh(
                    [j + (i - 0.5) * 0.34 for j, _ in data],
                    [v for _, v in data],
                    height=0.33,
                    color=color,
                    label=kind,
                )
            axes[0].set_yticks(
                range(len(labels)),
                [
                    "Reasoning",
                    "Calculator action",
                    "Financial reply",
                    "SQL error repair",
                    "Redirect lookup",
                    "Use valid lookup",
                ],
                fontsize=8,
            )
            axes[0].set_xlabel("Reduction in target NLL (nats/token)", fontsize=8)
            axes[0].axvline(0, color="black", linewidth=0.7)
            axes[0].invert_yaxis()
            axes[0].legend(frameon=False, fontsize=7)
            # Consecutive disjoint bins are weighted by the actual target count.
            chunks = [records[i : i + 24] for i in range(0, len(records), 24)]
            for key, label, c in (
                ("nll", "Target NLL", "#34666c"),
                ("parent_kl", "KL to parent", "#ad6846"),
            ):
                axes[1].plot(
                    [w[-1]["update"] for w in chunks],
                    [
                        sum(r[key] * r["positions"] for r in w)
                        / sum(r["positions"] for r in w)
                        for w in chunks
                    ],
                    label=label,
                    color=c,
                )
            axes[1].set_xlabel("Native optimizer update", fontsize=8)
            axes[1].set_ylabel("Nats/token", fontsize=8)
            axes[1].legend(frameon=False, fontsize=7)
            for ax in axes:
                ax.spines[["top", "right"]].set_visible(False)
            fig.savefig(P / "learner-prefix-fit.svg", metadata={"Date": None})
            plt.close(fig)
    reading = DOC / "learner-prefix-reading.json"
    if reading.is_file():
        committed(reading)
        root = RESULT / PRIMARY
        committed(root / "checksums.json")
        report["primary"] = read(root / "decision.json")
        report["reading"] = read(reading)
        records = []
        for arm in ("candidate", "control"):
            for r in unpack(root / (arm + ".json.gz")):
                q = r["requests"][0]
                t = q["trajectory"]
                records.append(
                    {
                        "id": r["id"],
                        "family": r["family"],
                        "arm": arm,
                        "success": q["score"]["success"],
                        "completed": q["score"]["completed"],
                        "wall_s": q["wall_s"],
                        "first_s": q["first_s"],
                        "peak": q["peak"],
                        "status": t["status"],
                        "answer": t["answer"],
                        "archive": digest(root / "checksums.json"),
                    }
                )
        write_csv(P / "learner-prefix-requests.csv", records)
    return report
