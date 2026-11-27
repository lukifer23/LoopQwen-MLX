"""Reconstruct the long-input result and final handoff figures from archived measurements."""

from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from loopqwen.io import digest, read
from scripts.assistance_deliberation import write_csv


def collect():
    root = Path("docs/measurements")
    run = read(root / "long-copy-fit/receipt.json")
    features = read(root / "long-copy-features.json")
    plan = read("research/diagnostics/long-copy-fit-runtime-02.json")
    assert run["plan"] == digest(
        Path("research/diagnostics/long-copy-fit-runtime-02.json")
    )
    assert run["weights"] == digest(
        root / "long-copy-fit/research-policy/step.safetensors"
    )
    assert (
        plan["features"] == digest(root / "long-copy-features.json")
        and features["passed"]
    )
    train, held = set(plan["train"]), set(plan["held"])
    assert len(train) == 80 and len(held) == 24 and train.isdisjoint(held)
    assert Counter((r["crate"], r["version"]) for r in run["history"]) == {
        (k, v): 4 for k in train for v in (0, 1)
    }
    result = {}
    raw = []
    for stage in ("parent", "one", "two"):
        rows = run["held"][stage]
        assert len(rows) == 48 and {(r["id"], r["version"]) for r in rows} == {
            (k, v) for k in held for v in (0, 1)
        }
        raw.extend(dict(stage=stage, **w) for w in rows)
        result[stage] = dict(
            correct=sum(
                all(r["margin"] > 0 for r in rows if r["id"] == k) for k in held
            ),
            nll=mean(r["nll"] for r in rows),
            address=mean(r["address"] for r in rows),
        )
    p = Path("paper")
    write_csv(p / "long-copy-held.csv", raw)
    write_csv(p / "long-copy-updates.csv", run["history"])
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), layout="constrained")
    stages = ("parent", "one", "two")
    for ax, metric, title in zip(
        axes,
        ("correct", "nll", "address"),
        (
            "Both source versions correct",
            "Answer loss (nats/token)",
            "Source-location loss (nats)",
        ),
        strict=True,
    ):
        values = [result[stage][metric] for stage in stages]
        bars = ax.bar(range(3), values, color=["#889da2", "#ae744c", "#34737c"])
        ax.bar_label(bars, fmt="%d" if metric == "correct" else "%.2f", padding=2)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 24 if metric == "correct" else max(values) * 1.18)
        ax.set_xticks(range(3), ["Parent", "One read", "Two reads"], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(p / "long-copy-held.svg", metadata={"Date": None})
    plt.close(fig)
    quality = read(root / "source-copy-supported-delivery.json")
    return dict(
        stages=result,
        passed=run["passed"],
        seconds=run["seconds"],
        peak=run["peak_bytes"],
        extraction_s=features["seconds"],
        prefixes=[w["histories"][0] for w in features["versions"]],
        held_nll=run["held"]["nll_gain"],
        held_address=run["held"]["address_gain"],
        depth=run["held"]["second_gain"],
        short=run["training"]["short_increase"],
        quality=quality,
    )
