"""Rebuild the paper's tables, figures, and study index from sealed native runs."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import digest, read

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "research/results"
LOOP, CONTROL = "#11675D", "#9D5932"
used = {}


def result(name):
    directory = RESULTS / name
    checks = read(directory / "checksums.json")
    registry = read(ROOT / "research/registry.json")["experiments"][name]
    if digest(directory / "checksums.json") != registry["archive"]:
        raise ValueError(name + " archive no longer matches registry")
    for file in ("decision.json", "registration.json"):
        if digest(directory / file) != checks[file]:
            raise ValueError(f"{name}/{file} failed verification")
    used[name] = registry["archive"]
    return read(directory / "decision.json")


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(PAPER / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def historical():
    arms = [
        ("Induction · primary", "native-induction-anchored-matched-01", 64),
        ("Induction · replication", "native-induction-anchored-replication-01", 64),
        ("Induction · second seed", "native-induction-anchored-seed2-01", 64),
        ("Deduction · replication", "deduction-replication-01", 56),
        ("Kinship · matched depth", "native-kinship-matched-01", 64),
        ("Arithmetic · pooled", "learned-exit-dense-01", 80),
        ("Entailment · pooled", "entailment-dense-01", 120),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for i, (label, study, n) in enumerate(arms):
        scores = result(study)["scores"]
        scores = scores.get("all", scores)
        vals = [scores[a]["correct"] / n * 100 for a in ("recurrent", "dense")]
        ax.plot(vals, [i, i], color="#9da9a7", lw=2, zorder=1)
        for x, c, m, d in zip(vals, [LOOP, CONTROL], ["o", "s"], [-0.07, 0.07]):
            ax.scatter(x, i + d, s=43, color=c, marker=m, zorder=3)
        ax.text(
            102,
            i,
            f"{scores['recurrent']['correct']} / {scores['dense']['correct']}  (n={n})",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(arms)), [t[0] for t in arms])
    ax.invert_yaxis()
    ax.set(xlim=(0, 132), xticks=range(0, 101, 20), xlabel="Correct responses (%)")
    ax.scatter([], [], color=LOOP, label="Repeated", marker="o")
    ax.scatter([], [], color=CONTROL, label="One pass", marker="s")
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, 1.18), ncol=2, frameon=False)
    finish(fig, "depth")
    timings = [
        (
            "Induction · Base 768",
            "native-induction-anchored-longbase-01",
            "recurrent",
            "base",
        ),
        ("Deduction · Base 768", "deduction-long-base-01", "recurrent", "base"),
        ("Kinship · matched", "native-kinship-matched-01", "recurrent", "dense"),
        ("Arithmetic · matched", "learned-exit-dense-01", "recurrent", "dense"),
        ("Entailment · matched", "entailment-dense-01", "recurrent", "dense"),
        ("Register · Base 128K", "conditional-language-128k", "candidate", "base"),
    ]
    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    for i, (label, name, a, b) in enumerate(timings):
        d = result(name)["scores"]
        d = d.get("all", d)
        ratio = d[a]["seconds"] / d[b]["seconds"]
        ax.barh(i, math.log2(ratio), color=LOOP if ratio < 1 else CONTROL, height=0.6)
        ax.text(
            1.03,
            i,
            f"{d[a]['seconds']:.1f} / {d[b]['seconds']:.1f} s",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(timings)), [t[0] for t in timings])
    ax.invert_yaxis()
    ax.set_xticks(
        [-5, -4, -3, -2, -1, 0, 1], ["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2"]
    )
    ax.set_xlim(-5.8, 3.4)
    ax.axvline(0, color="#343b39", lw=1)
    ax.set_xlabel("Repeated / control elapsed seconds (log scale)")
    finish(fig, "cost")


def index():
    registry = read(ROOT / "research/registry.json")
    with (PAPER / "registry-index.csv").open("w") as ledger:
        writer = csv.writer(ledger)
        writer.writerow(
            [
                "study",
                "registration_path",
                "registration_sha256",
                "question",
                "decision_rule",
                "max_wall_seconds",
                "archive_status",
            ]
        )
        sessions = registry.get("session_results", {})
        for name, entry in sorted(registry["experiments"].items()):
            path = Path("research/experiments") / (name + ".json")
            reg = read(ROOT / path)
            rule = reg.get("gate", reg.get("decision", "See registration"))
            if not isinstance(rule, str):
                rule = json.dumps(rule, sort_keys=True)
            archived = bool(entry["archive"]) or any(
                key.endswith("/" + name) for key in sessions
            )
            writer.writerow(
                [
                    name,
                    str(path),
                    entry["registration"],
                    reg.get("question", ""),
                    rule,
                    reg.get("limits", {}).get("wall_seconds", ""),
                    "archived" if archived else "no published measurements",
                ]
            )
    with (PAPER / "study-index.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            ["study", "question", "archive", "decision", "sha256_of_checksum_inventory"]
        )
        for name, v in sorted(registry["experiments"].items()):
            reg = read(ROOT / "research/experiments" / (name + ".json"))
            paths = [RESULTS / name]
            paths.extend(sorted((ROOT / "research/sessions").glob("*/" + name)))
            for path in paths:
                if not (path / "checksums.json").exists():
                    continue
                d = (
                    read(path / "decision.json")
                    if (path / "decision.json").exists()
                    else {}
                )
                w.writerow(
                    [
                        name,
                        reg.get("question", ""),
                        str(path.relative_to(ROOT)),
                        d.get("passed", d.get("status", "see archive")),
                        digest(path / "checksums.json"),
                    ]
                )


def training_lineage():
    columns = [
        "study",
        "training_parent",
        "backbone_revision",
        "shared_layer_start",
        "shared_passes",
        "rank",
        "prefix_bits",
        "trainable_parameters",
        "training_turns",
        "updates",
        "response_targets",
        "prepare_seconds",
        "optimizer_seconds",
        "training_sha256",
        "adapter_sha256",
    ]
    with (PAPER / "model-lineage.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for root in sorted(RESULTS.iterdir()):
            fit_path = root / "fit.json"
            if not fit_path.exists():
                continue
            fit = read(fit_path)
            if "turns" not in fit or "weights" not in fit:
                continue
            result(root.name)
            spec = read(root / "registration.json")
            donor = spec.get("donor", {})
            writer.writerow(
                [
                    root.name,
                    spec.get(
                        "initial_adapter", donor.get("repo", "Qwen/Qwen3.5-0.8B-Base")
                    ),
                    donor.get("revision", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
                    spec.get("start_layer", 23),
                    fit["passes"],
                    spec["rank"],
                    spec.get("prefix_bits", "BF16"),
                    fit["trainable"],
                    fit["turns"],
                    fit["updates"],
                    fit["targets"],
                    fit["prepare_seconds"],
                    fit["optimizer_seconds"],
                    fit["training"],
                    fit["weights"],
                ]
            )


def benchmark_table():
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.research_mixed_audit import audit

    review = ROOT / "docs/measurements/answer-audits"
    ent = read(review / "entailment-triplets-pooled.json")["scores"]["all"]["correct"]
    arithmetic = read(review / "learned-exit-pooled.json")["scores"]["all"]["content"]
    mixed = audit(
        "conditional-language-02", read(review / "conditional-language-02.json")
    )["scores"]["all"]["correct"]
    science = result("science-composition-01")["scores"]
    swaps = result("native-induction-class-swap-01")["both_correct"]
    long = result("conditional-language-128k")["scores"]
    live = result("tool-dialogue-boundary-01")["metrics"]
    external = result("tool-bfcl-transfer-01")["metrics"]
    rows = [
        ("Human inference, reviewed", ent["candidate"], ent["base"], 120, "Base"),
        (
            "Arithmetic, reviewed",
            arithmetic["candidate"],
            arithmetic["base"],
            80,
            "Base",
        ),
        ("Mixed language, reviewed", mixed["candidate"], mixed["base"], 80, "Base"),
        (
            "Science answer probe",
            science["recurrent"]["probe"],
            science["base"]["probe"],
            40,
            "Base",
        ),
        (
            "Native induction, whole class-swap pairs",
            swaps["recurrent"],
            swaps["dense"],
            32,
            "trained K1",
        ),
        (
            "Full-history register swap",
            long["candidate"]["correct"],
            long["base"]["correct"],
            2,
            "Base",
        ),
        (
            "Live catalog, corrected boundary",
            live["candidate"]["success"],
            live["control"]["success"],
            30,
            "Base",
        ),
        (
            "Published BFCL typed calls, catalog adapter",
            external["candidate"]["correct"],
            external["control"]["correct"],
            24,
            "Base",
        ),
    ]
    with (PAPER / "benchmarks.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "recurrent", "control", "n", "comparator"])
        writer.writerows(rows)


def main():
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.hashsalt": "loopqwen-paper",
            "svg.fonttype": "none",
        }
    )
    from paper_systems import write as system_comparisons

    system_comparisons(PAPER, RESULTS, result)
    benchmark_table()
    training_lineage()
    historical()
    tools_figure()
    trajectory_details()
    from paper_panels import generate

    generate(PAPER, RESULTS, result, finish)
    from paper_raw import make as plot_raw
    from paper_raw import selective

    from paper_field import make as plot_field

    plot_raw(PAPER, RESULTS, result, finish)
    selective(PAPER, RESULTS, result, finish)
    plot_field(PAPER, RESULTS, result, finish)
    from paper_table import make as table_actions

    table_actions(PAPER, RESULTS, result, finish)
    from paper_joint_tools import make as plot_joint_tools

    from paper_joint_tools import matched_effects

    plot_joint_tools(PAPER, RESULTS, result, finish)
    matched_effects(PAPER, RESULTS, result, finish)
    from paper_compute import make as compute_cost

    compute_cost(ROOT, PAPER)
    from paper_deployment import make as deployment
    from paper_deployment import transport

    from paper_conversations import make as conversations

    from paper_prefill import make as prefill

    from paper_prefill import resources as prefill_resources

    prefill(PAPER, RESULTS, result, finish)
    prefill_resources(PAPER, RESULTS, result, finish)
    conversations(PAPER, RESULTS, result)
    deployment(PAPER, RESULTS, result, finish)
    transport(PAPER, RESULTS, result, finish)
    index()
    (PAPER / "figure-sources.json").write_text(
        json.dumps(used, indent=2, sort_keys=True) + "\n"
    )


def tools_figure():
    from collections import defaultdict

    from loopqwen.evaluation.mechanism import unpack

    name = "tool-dialogue-boundary-01"
    if not (RESULTS / name / "decision.json").exists():
        return
    spec = read(RESULTS / name / "registration.json")
    result(name)
    items = read(ROOT / spec["material"])
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    labels = ["lookup", "chain", "compare", "direct", "clarify"]
    for side, offset, color in [("candidate", -0.17, LOOP), ("control", 0.17, CONTROL)]:
        path = RESULTS / name / (side + ".json.gz")
        if digest(path) != read(RESULTS / name / "checksums.json")[path.name]:
            raise ValueError("Tool record integrity")
        rows = unpack(path)
        counts = defaultdict(int)
        for r, w in zip(rows, items, strict=True):
            assert r["id"] == w["id"]
            counts[w["kind"]] += int(r["score"]["success"])
        xs = [i + offset for i in range(len(labels))]
        vals = [counts[k] for k in labels]
        bars = ax.bar(
            xs,
            vals,
            width=0.32,
            color=color,
            label="Trained loop" if side == "candidate" else "Untouched Base",
        )
        ax.bar_label(bars, padding=1, fontsize=8)
    ax.set_xticks(
        range(5),
        [
            "Lookup",
            "Two-call chain",
            "Compare via tools",
            "Direct answer",
            "Clarification",
        ],
    )
    ax.set(ylabel="Successful episodes / 6", ylim=(0, 7), yticks=range(0, 7, 2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
    finish(fig, "tools")


def trajectory_details():
    from loopqwen.evaluation.mechanism import unpack

    studies = [
        "tool-dialogue-boundary-01",
        "tool-dialogue-grounded-01",
        "tool-dialogue-typed-01",
    ]
    colors = ["#11675d", "#d69e4a", "#a94d3d", "#667784"]
    outcomes = ["Executed success", "Finished, incorrect", "Invalid call", "Incomplete"]
    bars = []
    with (PAPER / "tool-trajectories.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "study",
                "arm",
                "id",
                "family",
                "success",
                "valid_path",
                "correct_answer",
                "status",
                "executed_calls",
                "generated_tokens",
                "seconds",
            ]
        )
        for study in studies:
            if not (RESULTS / study / "checksums.json").exists():
                continue
            result(study)
            material = read(
                ROOT / read(RESULTS / study / "registration.json")["material"]
            )
            items = material["rows"] if isinstance(material, dict) else material
            for side in ("candidate", "control"):
                counts = [0, 0, 0, 0]
                path = RESULTS / study / (side + ".json.gz")
                if digest(path) != read(RESULTS / study / "checksums.json")[path.name]:
                    raise ValueError("Tool trajectory checksum mismatch")
                for r, item in zip(unpack(path), items, strict=True):
                    assert r["id"] == item["id"]
                    score = r["score"]
                    traj = r["trajectory"]
                    status = traj["status"]
                    counts[
                        0
                        if score["success"]
                        else 1
                        if status == "answered"
                        else 2
                        if status == "invalid_call"
                        else 3
                    ] += 1
                    family = item.get("kind") or item["function"][0]["name"]
                    writer.writerow(
                        [
                            study,
                            side,
                            item["id"],
                            family,
                            int(score["success"]),
                            int(score["path"]),
                            int(score["correct_answer"]),
                            status,
                            sum("observation" in t for t in traj["turns"]),
                            sum(len(t["token_ids"]) for t in traj["turns"]),
                            traj["seconds"],
                        ]
                    )
                short = {
                    "tool-dialogue-boundary-01": "Catalog",
                    "tool-dialogue-grounded-01": "Four-layer",
                    "tool-dialogue-typed-01": "Public execution",
                }[study]
                bars.append(
                    (
                        short + (" · recurrent" if side == "candidate" else " · Base"),
                        counts,
                    )
                )
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    for i, (name, counts) in enumerate(bars):
        total = sum(counts)
        loc = 0
        for j, (c, num) in enumerate(zip(colors, counts, strict=True)):
            fraction = 100 * num / total
            if num:
                ax.barh(i, fraction, left=loc, color=c, height=0.66)
                if fraction >= 8:
                    ax.text(
                        loc + fraction * 0.5,
                        i,
                        str(num),
                        ha="center",
                        va="center",
                        color="white" if j != 1 else "#232323",
                        fontsize=8,
                    )
            loc += fraction
    for col, label in zip(colors, outcomes, strict=True):
        ax.barh([], [], color=col, label=label)
    ax.set_yticks(range(len(bars)), [v[0] for v in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all episodes (%) · labels give counts")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    finish(fig, "trajectory-outcomes")


if __name__ == "__main__":
    main()
