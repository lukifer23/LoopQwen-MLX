"""Export every installed process and plot the four independent crossover pairs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read

ROOT = Path("research/results/extended-tool-prefix-01")
PAPER = Path("paper")


def build():
    rounds = unpack(ROOT / "rounds.json.gz")
    decision = read(ROOT / "decision.json")
    records = []
    for i, group in enumerate(rounds):
        for item in group["items"]:
            exchanges = item["exchanges"]
            turns = [turn for exchange in exchanges for turn in exchange["turns"]]
            records.append(
                {
                    "pair": i // 2 + 1,
                    "round": i + 1,
                    "mode": group["mode"],
                    "workflow": item["id"],
                    "requests": len(exchanges),
                    "turns": len(turns),
                    "output_tokens": sum(len(t["token_ids"]) for t in turns),
                    "reused_tokens": sum(t["reused_prompt_tokens"] for t in turns),
                    "seconds_to_open_turns": sum(t["ttft_s"] for t in turns),
                    "cold_seconds": item["cold_s"],
                    "cold_first_seconds": item["cold_first_s"],
                    "peak_bytes": item["peak"],
                }
            )
    with (PAPER / "extended-prefix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)
    for mode, official in decision["totals"].items():
        subset = [r for r in records if r["mode"] == mode]
        assert sum(r["output_tokens"] for r in subset) == official["output_tokens"]
        assert sum(r["reused_tokens"] for r in subset) == official["reused"]
        assert max(r["peak_bytes"] for r in subset) == official["peak"]
        assert (
            abs(
                sum(r["seconds_to_open_turns"] for r in subset)
                - official["turn_open_s"]
            )
            < 1e-7
        )
    metrics = ("turn_open_s", "cold_s", "cold_first_s", "peak", "decode_tps")
    labels = (
        "Native turn opening",
        "Cold response",
        "Cold first token",
        "Peak allocation",
        "Decode rate",
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.35), layout="constrained")
    for i, row in enumerate(decision["paired_ratios"]):
        ax.scatter(
            [(row[k] - 1) * 100 for k in metrics],
            [j + (i - 1.5) * 0.12 for j in range(len(metrics))],
            s=24,
            color=("#176b60", "#4b806e", "#55799e", "#796390")[i],
            label=f"Pair {i + 1}",
        )
    ax.axvline(0, color="#6c7374", lw=0.8)
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Change from fixed prefix (%)",
    )
    ax.invert_yaxis()
    ax.legend(ncols=4, fontsize=7, frameon=False, loc="lower left")
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(PAPER / "extended-prefix.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    build()
