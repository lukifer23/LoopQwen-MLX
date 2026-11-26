"""Report the transfer and retention strata of the single resident folded decoder."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    name = "folded-system-01"
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    d = result(name)
    all_groups = d["groups"]
    counts = list(all_groups["all"]["candidate"])
    labels = {
        "fresh": "Fresh human questions",
        "math": "Word problems",
        "judgment": "BoolQ",
        "public": "Numerical functions",
        "database": "SQLite",
        "file": "Project files",
        "package": "Installed packages",
        "all": "All requests",
    }
    with (paper / "folded-system.csv").open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "stratum",
                "side",
                *counts,
                "paired_gains",
                "paired_losses",
                "mcnemar_p",
                "cluster_p",
            ]
        )
        for key in labels:
            group = all_groups[key]
            for side in ("candidate", "control"):
                w.writerow(
                    [
                        key,
                        side,
                        *[group[side][k] for k in counts],
                        group["gains"],
                        group["losses"],
                        group["p"],
                        group["cluster_p"],
                    ]
                )
    checks = read(root / "checksums.json")
    subjects = read(read(root / "registration.json")["material"])["rows"]
    with (paper / "folded-requests.csv").open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "id",
                "family",
                "source",
                "side",
                "user_success",
                "correct_final",
                "completed",
                "grounded",
                "seconds",
                "first_token_s",
                "peak_bytes",
                "return_status",
                "answer",
            ]
        )
        for side in ("candidate", "control"):
            f = root / (side + ".json.gz")
            assert digest(f) == checks[f.name]
            for rec, item in zip(unpack(f), subjects, strict=True):
                assert rec["id"] == item["id"]
                scores = rec["score"]
                w.writerow(
                    [
                        rec["id"],
                        item["family"],
                        item["source"],
                        side,
                        *[
                            scores[k]
                            for k in ("success", "answer", "completed", "grounded")
                        ],
                        rec["wall_s"],
                        rec["first_token_s"],
                        rec["peak"],
                        rec["trajectory"]["status"],
                        rec["trajectory"]["answer"],
                    ]
                )
    fig, (score, latency) = plt.subplots(1, 2, figsize=(7.3, 3.3), sharey=True)
    for i, (key, label) in enumerate(labels.items()):
        g = all_groups[key]
        for side, direction, color, marker, title in (
            ("candidate", -0.10, "#11675d", "o", "Folded"),
            ("control", 0.10, "#9d5932", "s", "Untouched Base"),
        ):
            v = g[side]
            opts = {
                "marker": marker,
                "color": color,
                "s": 28,
                "label": title if i == 0 else None,
            }
            score.scatter(v["success"] / v["n"] * 100, i + direction, **opts)
            latency.scatter(v["seconds"] / v["n"], i + direction, **opts)
    score.set_yticks(range(len(labels)), labels.values())
    score.invert_yaxis()
    score.set_xlabel("Correct user outcomes (%)")
    score.set_xlim(-4, 104)
    latency.set_xlabel("Measured seconds per request")
    latency.set_xlim(left=0)
    score.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(-0.07, 1), ncol=2, fontsize=8
    )
    finish(fig, "folded-system")
