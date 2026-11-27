"""Rebuild the measured recurrent-state and supervision study from immutable native evidence."""

from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv

P = Path("paper")
R = Path("research/results/conditional-features-01")
DOC = Path("docs/measurements")
GROUPS = {
    "instruction": "Human instructions",
    "arithmetic": "Worked arithmetic",
    "legacy": "Earlier assistance",
    "explanation": "Natural table explanations",
    "program": "Targeted SQL",
    "discovery": "Schema calls",
    "scalar": "Answers after scalar reads",
}


def build():
    spec = read(R / "registration.json")
    result = read(R / "decision.json")
    assert result["passed"]
    audits = read(DOC / "conditional-native-feature-replay.json")
    assert audits["feature_archive"] == digest(R / "checksums.json")
    cold = [q for r in audits["records"] for q in r["left_context_checks"]]
    actual = unpack(R / "feature-index.json.gz")
    fit_spec = Path("research/experiments/conditional-program-fit-01.json")
    exposure = read(DOC / "conditional-planned-exposure.json")
    assert exposure["registration"] == digest(fit_spec)
    assert sum(r["positions"] for r in actual) == result["positive_positions"]
    jobs = [
        j
        for p in Path("research/sessions").glob("window-*.json")
        for j in read(p)["jobs"]
        if j["job"] == spec["id"]
    ]
    assert jobs
    by_group = []
    for group, label in GROUPS.items():
        g = [r for r in actual if r["group"] == group]
        planned = exposure["by_group"][group]
        positions = sum(r["positions"] for r in g)
        assert (
            positions == planned["distinct_position_instances"]
            and len(g) == planned["source_turns"]
        )
        by_group.append(
            {
                "group": group,
                "source": label,
                "examples": len(g),
                "initial_tokens": sum(r["prefix_tokens"] for r in g),
                "positions": positions,
                "presentations": planned["presentations"],
                "supervised": positions * planned["presentations"] // len(g),
                "alternate_positions": sum(
                    len(n["negative_positions"]) for r in g for n in r["negative"]
                ),
            }
        )
    assert (
        sum(g["supervised"] for g in by_group)
        == exposure["supervised_target_positions"]
    )
    write_csv(P / "conditional-feature-sources.csv", by_group)
    write_csv(
        P / "conditional-left-context-checks.csv",
        [
            {"id": r["turn"], "group": r["group"], "edit": r["edit"], **q}
            for r in audits["records"]
            for q in r["left_context_checks"]
        ],
    )
    distance = [r["rms_relative"] for r in cold]
    info = {
        "archive": digest(R / "checksums.json"),
        "backbone": read(R / "backbone.json"),
        "groups": by_group,
        "parity": {k: v for k, v in result["parity"].items() if k != "prefix"},
        "initial_test_tokens": len(result["parity"]["prefix"]),
        "feature_bytes": result["feature_bytes"],
        "positive_positions": result["positive_positions"],
        "negative_positions": result["negative_positions"],
        "worker_seconds": sum(j["seconds"] for j in jobs),
        "peak_gib": max(j["peak_bytes"] for j in jobs) / 2**30,
        "replay_turns": len(audits["records"]),
        "replay_positions": sum(r["regenerated_positions"] for r in audits["records"]),
        "left_checks": len(cold),
        "left_top_matches": sum(r["left_top"] == r["cached_top"] for r in cold),
        "relative_rms_median": median(distance),
        "relative_rms_max": max(distance),
        "left_nll_max": max(abs(r["left_nll"] - r["cached_nll"]) for r in cold),
        "planned_channels": exposure["supervision_by_channel"],
        "planned_updates": exposure["native_updates_planned"],
        "trace_group_counts": dict(Counter(r["group"] for r in audits["records"])),
    }
    save(P / "conditional-feature-study.json", info)
    plt.rcParams.update(
        {
            "svg.hashsalt": "conditional-feature-01",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.5, 2.9),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    yy = list(range(len(by_group)))
    left.barh(
        [j + 0.17 for j in yy],
        [r["supervised"] / 1000 for r in by_group],
        height=0.32,
        color="#236a60",
        label="Planned fit labels",
    )
    left.barh(
        [j - 0.17 for j in yy],
        [r["positions"] / 1000 for r in by_group],
        height=0.32,
        color="#a85a30",
        label="Measured positions",
    )
    left.set(
        yticks=yy,
        yticklabels=[r["source"] for r in by_group],
        xlabel="Thousands of tokens",
    )
    left.invert_yaxis()
    left.legend(frameon=False, fontsize=8)
    right.bar(
        ["GDN 20", "GDN 21", "GDN 22", "Attn 23"],
        info["parity"]["suffix_state_changes"],
        color="#236a60",
    )
    right.set(
        ylabel="Mean absolute change in hidden state",
        xlabel="Layer on second physical visit",
    )
    right.tick_params(axis="x", labelrotation=24)
    fig.savefig(P / "conditional-feature-study.svg", metadata={"Date": None})
    plt.close(fig)
    dest = P / "conditional-feature-study.pdf"
    dest.write_bytes(
        typst.compile(str(P / "conditional-feature-study.typ"), timestamp=0)
    )
    print(info["replay_positions"], info["left_checks"], digest(dest))


if __name__ == "__main__":
    build()
