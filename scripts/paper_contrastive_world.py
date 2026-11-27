"""Publish only closed, verified stages of the evidence-changing Qwen experiment."""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv
from scripts.conditional_deliberation import account as channel_account
from scripts.contrastive_fit_specificity import run as source_specificity
from scripts.table_policy_reading import committed
from scripts.wording_sensitivity import run as sensitivity

P = Path("paper")
DOC = Path("docs/measurements")
RESULT = Path("research/results")
SOURCE = Path("research/sources/contrastive-worlds")
STAGES = (
    "contrastive-table-features-01",
    "contrastive-language-features-01",
    "contrastive-world-fit-01",
    "contrastive-world-transfer-01",
)
NAMES = (
    "Table extraction",
    "Reasoning and service extraction",
    "Recurrent fit",
    "Held product comparison",
)


def overview():
    seen = []
    exposure = []
    windows = {}
    for path in Path("research/sessions").glob("window-*.json"):
        for j in read(path)["jobs"]:
            if j["job"] in STAGES:
                assert j["job"] not in windows
                windows[j["job"]] = (j, path)
    for i, name in enumerate(STAGES):
        root = RESULT / name
        if not (root / "checksums.json").exists():
            break
        committed(root / "checksums.json")
        if i == 3 and (root / "control.json.gz").is_file():
            for kind in ("services", "qualitative"):
                committed(DOC / ("contrastive-" + kind + "-reading.json"))
        d = read(root / "decision.json")
        s = read(root / "registration.json")
        j, window = windows[name]
        row = {
            "id": name,
            "name": NAMES[i],
            "passed": d["passed"],
            "elapsed": j["seconds"],
            "peak_gib": j["peak_bytes"] / 2**30,
            "max_seconds": s["limits"]["wall_seconds"],
            "archive": digest(root / "checksums.json"),
            "window": digest(window),
            "decision": d,
        }
        cold = DOC / (name + "-cold.json")
        if cold.is_file():
            committed(cold)
            receipt = read(cold)
            row["cold"] = digest(cold)
            if "full_inventory" in receipt:
                inv = receipt["full_inventory"]
                row["checked_files"] = inv["checked_files"]
                row["checked_bytes"] = inv["checked_bytes"]
                row["rebuilt"] = len(inv["regenerated_full_files"])
                row["native_cases"] = len(receipt["cases"])
            elif i == 2:
                row["probe_positions"] = receipt["positions"]
                row["max_mean_error"] = receipt["max_mean_error"]
        seen.append(row)
        if i < 2:
            path = root / (
                "feature-index.json.gz"
                if (root / "feature-index.json.gz").exists()
                else "partial-index.json.gz"
            )
            groups = defaultdict(list)
            for item in unpack(path):
                groups[item["group"]].append(item)
            for group, items in sorted(groups.items()):
                exposure.append(
                    {
                        "group": group,
                        "turns": len(items),
                        "target_positions": sum(r["positions"] for r in items),
                        "opening_tokens": sum(r["prefix_tokens"] for r in items),
                        "negative_positions": sum(
                            len(z["negative_positions"])
                            for r in items
                            for z in r["negative"]
                        ),
                        "feature_bytes": sum(r["bytes"] for r in items),
                    }
                )
        if not d["passed"]:
            break
    if not seen:
        raise ValueError(
            "A verified native stage is required for an experimental report"
        )
    return seen, exposure


def held(out):
    root = RESULT / STAGES[-1]
    panel = unpack(SOURCE / "transfer-panel.json.gz")["rows"]
    runs = [unpack(root / (arm + ".json.gz")) for arm in ("candidate", "control")]
    rows = []
    for w, a, b in zip(panel, *runs, strict=True):
        assert w["id"] == a["id"] == b["id"]
        r = {
            "id": w["id"],
            "family": w["family"],
            "cluster": w["cluster"],
            "opening": len(w["initial_ids"]),
        }
        for arm, record in (("loop", a), ("base", b)):
            (q,) = record["requests"]
            tr = q["trajectory"]
            assert tr["turns"][0]["prompt_ids"] == w["initial_ids"]
            r.update(
                {
                    arm + "_" + k: v
                    for k, v in {
                        "success": q["score"]["success"],
                        "completed": q["score"]["completed"],
                        "wall_s": q["wall_s"],
                        "first_s": q["first_s"],
                        "peak": q["peak"],
                        "output_tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                        "visible": tr["answer"] or "",
                    }.items()
                }
            )
        rows.append(r)
    write_csv(P / "contrastive-world-requests.csv", rows)
    base_features = read("research/experiments/conditional-features-01.json")
    out["generation_channels"] = channel_account(
        root,
        runs,
        destination=P / "contrastive-world-generation-channels.csv",
        additional_training=(Path(base_features["training"]),),
        reasoning_families=("qualitative",),
    )
    sensitivity("contrastive-worlds", "report")
    out["sensitivity"] = read(DOC / "contrastive-worlds-wording-sensitivity.json")[
        "cells"
    ]
    out["service"] = read(DOC / "contrastive-services-reading.json")["source_pairs"]
    out["qualitative"] = read(DOC / "contrastive-qualitative-reading.json")


def plot(rows):
    plt.rcParams.update({"svg.hashsalt": "contrastive-exposures", "font.size": 9})
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(9, 3.4), layout="constrained", sharey=True
    )
    labels = [r["group"].replace("_", " ") for r in rows]
    a.barh(labels, [r["target_positions"] / 1000 for r in rows], color="#306f77")
    b.barh(labels, [r["negative_positions"] / 1000 for r in rows], color="#ad5337")
    a.set_xlabel("Positive label positions (thousands)")
    b.set_xlabel("Changed negative positions (thousands)")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(P / "contrastive-world-exposure.svg", metadata={"Date": None})
    plt.close(fig)


def optimizer_plot(log):
    totals = []
    positions = 0
    for step in range(0, len(log), 64):
        block = log[step : step + 64]
        n = sum(r["targets"] for r in block)
        pairs = sum(r["preferences"] for r in block)
        positions += n
        assert n > 0
        totals.append(
            {
                "through_update": block[-1]["step"],
                "positions": positions,
                "mean_nll": sum(r["nll_sum"] for r in block) / n,
                "mean_teacher_kl": sum(r["teacher_kl_sum"] for r in block) / n,
                "mean_preference_loss": sum(r["preference_loss_sum"] for r in block)
                / pairs
                if pairs
                else None,
            }
        )
    plt.rcParams.update({"svg.hashsalt": "contrastive-optimizer", "font.size": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.4), layout="constrained")
    for ax, (key, label) in zip(
        axes,
        (
            ("mean_nll", "Response NLL per token"),
            ("mean_teacher_kl", "Teacher KL per token"),
            ("mean_preference_loss", "Softplus loss per preference"),
        ),
        strict=True,
    ):
        ax.plot(
            [r["positions"] / 1e6 for r in totals],
            [r[key] for r in totals],
            color="#306f77",
            lw=1.6,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Cumulative label exposures (M)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(P / "contrastive-world-optimization.svg", metadata={"Date": None})
    plt.close(fig)
    write_csv(P / "contrastive-world-optimization.csv", totals)


def build():
    seen, exposure = overview()
    spec = read("research/experiments/" + STAGES[2] + ".json")
    env = read(SOURCE / "transfer-environment-replay.json")
    out = {
        "stages": seen,
        "exposures": exposure,
        "stage_count": len(seen),
        "limits": spec["limits"],
        "fit_target": spec["expected_positions"],
        "preference_target": spec["expected_preferences"],
        "latest": seen[-1],
        "enrollment": read(SOURCE / "independent-replay.json"),
        "held_environment": env,
        "corpus": digest(SOURCE / "training.json.gz"),
        "total_native_s": sum(r["elapsed"] for r in seen),
        "source_words": read(
            "research/diagnostics/contrastive-worlds-wording-strata.json"
        )["counts"],
    }
    if len(seen) >= 3:
        root = RESULT / STAGES[2]
        if (root / "training-log.json.gz").exists():
            log = unpack(root / "training-log.json.gz")
            if log:
                # Logged scheduled updates contain sums/counts or normalized diagnostics
                # according to the original objective. Preserve the original units.
                write_csv(P / "contrastive-world-optimizer.csv", log)
                optimizer_plot(log)
        if (root / "training-free-generation.json.gz").is_file():
            out["specificity"] = source_specificity(True)
    if len(seen) == 4 and (RESULT / STAGES[-1] / "control.json.gz").is_file():
        held(out)
    if exposure:
        plot(exposure)
        write_csv(P / "contrastive-world-exposure.csv", exposure)
    save(P / "contrastive-world.json", out)
    pdf = P / "contrastive-world.pdf"
    typst.compile(
        str(P / "contrastive-world.typ"), output=str(pdf), root=".", timestamp=0
    )
    doc = PdfReader(pdf)
    assert all(len(p.extract_text()) > 180 for p in doc.pages)
    print(len(seen), "archived stages,", len(doc.pages), "pages")


if __name__ == "__main__":
    build()
