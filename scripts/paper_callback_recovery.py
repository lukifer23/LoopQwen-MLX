"""User-visible outcomes after genuine calculator exceptions on unseen questions."""

import csv
import json

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read


def make(paper, results, result, finish):
    name = "callback-recovery-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    attempts = unpack(root / "opening-attempts.json.gz")
    with (paper / "calculator-training-attempts.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "training_id",
                "opening_status",
                "submitted_call",
                "observed_result",
                "eligible",
            ]
        )
        for a in attempts:
            tr = a["trajectory"]
            turn = tr["turns"][0]
            writer.writerow(
                [
                    a["id"],
                    tr["status"],
                    turn.get("visible"),
                    json.dumps(turn.get("observation")),
                    int("error" in turn.get("observation", {})),
                ]
            )
    if "groups" not in decision:
        return
    panel = read(read(root / "registration.json")["material"])["rows"]
    with (paper / "calculator-recovery-requests.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "cohort",
                "family",
                "arm",
                "first_seconds",
                "wall_seconds",
                "peak_mlx_bytes",
                "terminal",
                "final_answer",
                "correct_final_answer",
                "successful_grounded",
                "observations",
                "recovery_turns",
            ]
        )
        for arm in ("candidate", "control"):
            for w, r in zip(panel, unpack(root / (arm + ".json.gz")), strict=True):
                t = r["trajectory"]
                writer.writerow(
                    [
                        r["id"],
                        w["cohort"],
                        w["family"],
                        arm,
                        r["first_s"],
                        r["wall_s"],
                        r["peak"],
                        t["status"],
                        t["answer"],
                        r["score"]["answer"],
                        r["score"]["grounded"],
                        json.dumps(
                            [v["observation"] for v in t["turns"] if "observation" in v]
                        ),
                        sum(v.get("recovery_active", False) for v in t["turns"]),
                    ]
                )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, name in zip(axes, ("ASDiv", "SVAMP"), strict=True):
        g = decision["groups"][name]
        for side, offset, label, color in (
            ("candidate", -0.18, "Recurrent", "#11675d"),
            ("control", 0.18, "Base", "#9d5932"),
        ):
            vals = [
                g[side][field]
                for field in ("answer", "grounded", "successful_after_error")
            ]
            ax.bar(
                [x + offset for x in range(3)],
                vals,
                width=0.34,
                color=color,
                label=label,
            )
        ax.set_xticks(
            [0, 1, 2],
            ["Correct\nanswer", "Correct\nwith call", "Correct after\nexception"],
        )
        ax.set_ylim(0, g["n"] + 2)
        ax.set_title(name)
    axes[0].set_ylabel("Requests (of 24)")
    axes[1].legend(frameon=False, fontsize=8)
    finish(fig, "calculator-recovery")
