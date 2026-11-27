"""Publish the actual four-layer Base fit, its native cache and the reloaded likelihood checks."""

import csv
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typst
from pypdf import PdfReader

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv

P = Path("paper")
R = Path("research/results/table-reentry-fit-01")
D = Path("docs/measurements")
GROUPS = ("changed", "database", "language", "earlier")
LABELS = {
    "changed": "Changed tables",
    "database": "Database dialogue",
    "language": "Open and reading",
    "earlier": "Earlier tools",
}
COLORS = ("#166851", "#a4632e", "#765487", "#496e96")


def build():
    s = read(R / "registration.json")
    fit = read(R / "fit.json")
    decision = read(R / "decision.json")
    index = read(R / "checksums.json")
    replay = read(D / "table-reentry-fit-native-replay.json")
    cache = read(D / "table-reentry-feature-cache.json")
    backbone = read(D / "table-reentry-backbone.json")
    exposure = read(D / "reentry-target-exposure.json")
    assert exposure["rows"] == digest(P / "reentry-target-exposure.csv")
    assert replay["weights"] == index["reentry.safetensors"] == fit["weights"]
    assert (
        replay["backbone"]
        == backbone["native_pack_sha256"]
        == fit["backbone"]["sha256"]
    )
    assert cache["rows"] == digest(P / "reentry-feature-cache.csv")
    with (P / "reentry-feature-cache.csv").open(newline="") as f:
        features = {r["view"]: r for r in csv.DictReader(f)}
    log = unpack(R / "optimizer-steps.json.gz")
    draws = []
    steps = []
    for r in log:
        steps.append({"step": r["step"], "unclipped_gradient_norm": r["gradient_norm"]})
        for v in r["observations"]:
            f = features[v["view"]]
            draws.append(
                {
                    "step": r["step"],
                    "group": f["group"],
                    "view": v["view"],
                    "sequence_tokens": int(f["sequence_length"]),
                    "target_positions": len(v["positions"]),
                    "mean_loss": v["loss"],
                }
            )
    assert len(draws) == 4 * s["warm_updates"] and {r["group"] for r in draws} == set(
        GROUPS
    )
    assert exposure["registration"] == digest(R / "registration.json")
    assert exposure["selected_positions"] == sum(r["target_positions"] for r in draws)
    assert exposure["causal_input_tokens"] == sum(r["sequence_tokens"] for r in draws)
    write_csv(P / "reentry-training-draws.csv", draws)
    write_csv(P / "reentry-training-gradients.csv", steps)
    probs = read(R / "training-probes.json")
    recovered = {r["view"]: r for r in replay["views"]}
    evidence = []
    for r in probs:
        recorded = recovered[r["view"]]
        evidence.append(
            {
                "group": r["group"],
                "view": r["view"],
                "targets": len(r["positions"]),
                "first_nll": r["first_nll"],
                "second_nll": r["second_nll"],
                "gain": r["first_nll"] - r["second_nll"],
                "first_replay_error": recorded["first_abs_error"],
                "second_replay_error": recorded["second_abs_error"],
            }
        )
    write_csv(P / "reentry-fit-probes.csv", evidence)
    plt.rcParams.update(
        {
            "svg.hashsalt": "reentry-base-fit",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.3, 1]},
    )
    summary = []
    for i, (g, color) in enumerate(zip(GROUPS, COLORS, strict=True)):
        group = [r for r in draws if r["group"] == g]
        assert len(group) == s["warm_updates"]
        bins = [group[j : j + 64] for j in range(0, len(group), 64)]
        left.plot(
            [mean(r["step"] for r in batch) for batch in bins],
            [mean(r["mean_loss"] for r in batch) for batch in bins],
            color=color,
            label=LABELS[g],
        )
        b = [r for r in evidence if r["group"] == g]
        assert len(b) == 8
        for j, r in enumerate(b):
            right.plot(
                [i - 0.18, i + 0.18],
                [r["first_nll"], r["second_nll"]],
                color=color,
                alpha=0.47,
                lw=0.7,
            )
        a1, a2 = mean(r["first_nll"] for r in b), mean(r["second_nll"] for r in b)
        right.plot(
            [i - 0.18, i + 0.18], [a1, a2], color=color, lw=2.2, marker="o", ms=3
        )
        summary.append(
            [
                LABELS[g],
                len({r["view"] for r in group}),
                sum(r["target_positions"] for r in group),
                f"{a1:.3f}",
                f"{a2:.3f}",
                sum(r["gain"] > 0 for r in b),
            ]
        )
    left.set(
        xlabel="Native optimizer update", ylabel="Training token loss (64-update mean)"
    )
    left.legend(frameon=False, fontsize=6.7)
    right.set(
        xticks=np.arange(4),
        xticklabels=["Tables", "SQL", "Language", "Tools"],
        ylabel="Gold-token NLL on training probes",
    )
    fig.savefig(P / "reentry-fit.svg", metadata={"Date": None})
    plt.close(fig)
    parity = [read(R / (p + "-parity.json")) for p in ("initial", "trained")]
    jobs = [
        j
        for f in Path("research/sessions").glob("window-*.json")
        for j in read(f)["jobs"]
        if j["job"] == s["id"]
    ]
    assert jobs and len(probs) == 32 and len(features) == fit["views"]
    report = {
        "passed": decision["passed"],
        "groups": summary,
        "views": fit["views"],
        "targets": sum(r["target_positions"] for r in draws),
        "tokens": sum(r["sequence_tokens"] for r in draws),
        "steps": len(steps),
        "qualified_gates": [
            [k, "Pass" if v else "Miss"] for k, v in decision["gates"].items()
        ],
        "mean_gain": f"{decision['training_mean_nll_gain']:.4f}",
        "probe_targets": replay["positions"],
        "probe_wins": sum(r["gain"] > 0 for r in evidence),
        "max_replay_error": f"{replay['max_abs_nll_error']:.8f}",
        "positive_gradient_steps": sum(r["unclipped_gradient_norm"] > 0 for r in steps),
        "clipped_steps": sum(r["unclipped_gradient_norm"] > 1 for r in steps),
        "grad_median": f"{median(r['unclipped_gradient_norm'] for r in steps):.4f}",
        "gate_rms": f"{fit['gate_rms']:.4f}",
        "up_norms": [f"{v:.3f}" for v in fit["residual_norms"]],
        "trained": fit["trainable"],
        "weights": fit["weights"],
        "cold_exact": read(R / "packed-base.json")["compared"],
        "cache_seconds": round(cache["elapsed_first_to_last_s"], 2),
        "cache_tokens": cache["positions"],
        "cache_bytes": cache["stored_bytes"],
        "window_native_s": round(sum(j["seconds"] for j in jobs), 2),
        "mlx_peak_gib": round(max(j["peak_bytes"] for j in jobs) / 2**30, 3),
        "parity": [
            [
                name,
                r["full_top"],
                r["incremental_top"],
                f"{r['max_logit_error']:.4f}",
                r["attention_offsets"][0],
                r["unique_slots"],
            ]
            for name, r in zip(("Initial", "Trained"), parity, strict=True)
        ],
        "opt_digest": index["optimizer.safetensors"],
        "archive": digest(R / "checksums.json"),
    }
    save(P / "reentry-fit-paper.json", report)
    pdf = typst.compile(str(P / "reentry-fit.typ"), timestamp=0)
    path = P / "reentry-fit.pdf"
    path.write_bytes(pdf)
    assert pdf == typst.compile(str(P / "reentry-fit.typ"), timestamp=0)
    pages = PdfReader(path).pages
    assert all(len(p.extract_text()) > 350 for p in pages)
    print(len(pages), "pages", digest(path), flush=True)


if __name__ == "__main__":
    build()
