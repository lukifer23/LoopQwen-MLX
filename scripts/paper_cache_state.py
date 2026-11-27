"""Typeset the two preregistered cache experiments from their original native traces."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from scripts.cache_state_conversations import NAME, summary, verify

P = Path("paper")
R = Path("research/results")
NAMES = {
    "packages": "Installed versions",
    "file": "File size and lines",
    "agenda": "Conference schedule",
    "wheel": "Release manifest",
}


def run():
    installed = R / NAME
    d = read(installed / "decision.json")
    verify(installed, d)
    rounds = unpack(installed / "rounds.json.gz")
    kernel = read(R / "cache-state-prefill-01/decision.json")
    assert kernel["passed"] and kernel["gates"]["numeric"]
    records = []
    for i, block in enumerate(rounds):
        for r in block["items"]:
            q = summary([r])
            records.append(
                {
                    "round": i + 1,
                    "setting": block["mode"],
                    "workload": r["id"],
                    "cold_s": r["cold_s"],
                    "first_s": r["cold_first_s"],
                    "peak_bytes": r["peak"],
                    **{
                        k: q[k]
                        for k in (
                            "turn_open_s",
                            "decode_tps",
                            "output_tokens",
                            "reused",
                            "state_blocks",
                        )
                    },
                    "exchanges": len(r["exchanges"]),
                    "completed": sum(x["status"] == "answered" for x in r["exchanges"]),
                }
            )
    with (P / "cache-state-conversations.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator="\n")
        out.writeheader()
        out.writerows(records)
    plt.rcParams.update(
        {
            "svg.hashsalt": "loopqwen-cache-2026",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.45), layout="constrained")
    for j, (title, ratios, color, mark) in enumerate(
        (
            ("Fixed prompts", kernel["first_ratios"], "#29736b", "o"),
            ("Installed conversations", d["native_open_ratios"], "#6c568c", "s"),
        )
    ):
        xs = [i + 1 + (j - 0.5) * 0.075 for i in range(4)]
        ax.plot(
            xs,
            [100 * (1 - v) for v in ratios],
            marker=mark,
            color=color,
            label=title,
            linewidth=1.5,
        )
    ax.axhline(0, color="#858d8c", linewidth=0.8)
    ax.set(
        xticks=[1, 2, 3, 4],
        xlabel="Crossover pair in execution order",
        ylabel="Native opening time saved (%)",
        ylim=(-0.3, 5),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncols=2, frameon=False)
    fig.savefig(P / "cache-state-conversations.svg", metadata={"Date": None})
    plt.close(fig)
    table = []
    for family, label in NAMES.items():
        vals = []
        for mode in ("stock", "state"):
            selected = [
                r
                for block in rounds
                if block["mode"] == mode
                for r in block["items"]
                if r["id"] == family
            ]
            q = summary(selected)
            vals.extend([f"{q['cold_s']:.2f}", f"{q['turn_open_s']:.3f}"])
        table.append([label, *vals])
    metrics = []
    for name, k, factor, precision in (
        ("Complete process (s)", "cold_s", 1, 3),
        ("First generated token from launch (s)", "cold_first_s", 1, 3),
        ("All native turn openings (s)", "turn_open_s", 1, 3),
        ("Decode (tokens/s)", "decode_tps", 1, 3),
        ("Peak native allocation (MiB)", "peak", 2**-20, 2),
        ("Generated tokens", "output_tokens", 1, 0),
        ("Reused input tokens", "reused", 1, 0),
    ):
        metrics.append(
            [
                name,
                *[
                    f"{d['totals'][side][k] * factor:,.{precision}f}"
                    for side in ("stock", "state")
                ],
            ]
        )
    save(
        P / "cache-state-runtime.json",
        {
            "metrics": metrics,
            "by_workload": table,
            "wheel": read(installed / "installed-wheel.json"),
        },
    )
    lines = typst.compile(str(P / "cache-state-runtime.typ"), timestamp=0)
    (P / "cache-state-runtime.pdf").write_bytes(lines)
    doc = PdfReader(P / "cache-state-runtime.pdf")
    assert len(doc.pages) == 2 and all(len(p.extract_text()) > 500 for p in doc.pages)


if __name__ == "__main__":
    run()
