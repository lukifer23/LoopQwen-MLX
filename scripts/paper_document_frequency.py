"""Plot actual source evidence positions and measured cold costs across whole documents."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read


def make(paper, results, result, finish):
    study = "document-frequency-reader-01"
    root = results / study
    result(study)
    panel = read(read(root / "registration.json")["material"])["panel"]
    data = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    fig, (positions, times) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), width_ratios=(1.4, 1)
    )
    colors = {"candidate": "#11675d", "control": "#9d5932"}
    labels = []
    with (paper / "document-frequency-reader.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "format",
                "question",
                "arm",
                "input_tokens",
                "gold_byte_count",
                "gold_start",
                "gold_stop",
                "selected_start",
                "selected_stop",
                "answer",
                "completed_exact",
                "content",
                "cold_first_s",
                "cold_wall_s",
                "peak_bytes",
                "output_tokens",
                "prefill_attended_fraction",
            ]
        )
        for i, w in enumerate(panel):
            label = (w["id"][:7] + " tree") if w["family"] == "tree" else "uv.lock"
            labels.append(label)
            n = len(w["prompt_ids"])
            positions.hlines(i, 0, n / 1000, color="#bac1c2", lw=3)
            positions.scatter(
                w["gold_start"] / 1000,
                i,
                marker="o",
                facecolors="white",
                edgecolors="#161c1b",
                s=68,
                label="Reference" if i == 0 else None,
                zorder=2,
            )
            for side in data:
                r = data[side][i]
                sec = r.get("attention_memory", {})
                if sec and side == "candidate":
                    positions.scatter(
                        (sec["start"] + sec["stop"]) / 2000,
                        i,
                        s=30,
                        color=colors[side],
                        marker="x",
                        label="Selected" if i == 0 else None,
                        zorder=3,
                    )
                shift = -0.17 if side == "candidate" else 0.17
                bar = times.barh(
                    i + shift,
                    r["cold_seconds"],
                    height=0.3,
                    color=colors[side],
                    label=("Loop" if side == "candidate" else "Base")
                    if i == 0
                    else None,
                )
                times.bar_label(
                    bar, labels=[f"{r['cold_seconds']:.0f}"], padding=2, fontsize=7
                )
                writer.writerow(
                    [
                        w["id"],
                        w["family"],
                        w["question"],
                        side,
                        n,
                        w["gold"],
                        w["gold_start"],
                        w["gold_stop"],
                        sec.get("start"),
                        sec.get("stop"),
                        r["grade"]["visible"],
                        r["grade"]["exact"],
                        r["grade"]["content"],
                        r["cold_first"],
                        r["cold_seconds"],
                        r["peak_bytes"],
                        len(r["token_ids"]),
                        (
                            sum(s["attended"] for s in r["page_sites"])
                            / sum(s["possible"] for s in r["page_sites"])
                        )
                        if r.get("page_sites")
                        else "",
                    ]
                )
    positions.set_yticks(range(len(panel)), labels)
    times.set_yticks(range(len(panel)), [""] * len(panel))
    positions.set(
        xlabel="Position in complete input (thousands of tokens)",
        title="Evidence in the original document",
    )
    times.set(xlabel="Seconds", title="Cold request time")
    times.margins(x=0.18)
    for ax in (positions, times):
        ax.invert_yaxis()
        ax.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=2,
            frameon=False,
        )
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "document-frequency-reader")
