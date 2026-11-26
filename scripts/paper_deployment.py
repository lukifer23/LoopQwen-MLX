"""Recount a shipped assistant's cold and mixed-mode field workloads."""

import csv
import json

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.evaluation.statistics import exact_mcnemar
from loopqwen.io import digest, read
from loopqwen.research.mixed_field import outcome


def make(paper, results, result, finish):
    name = "release-mixed-field-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    checks = read(root / "checksums.json")
    data = {}
    for side in ("candidate", "control"):
        source = root / (side + ".json.gz")
        if digest(source) != checks[source.name]:
            raise ValueError("Field transcript changed")
        data[side] = unpack(source)
    if outcome(data, read(root / "registration.json")) != decision:
        raise ValueError("Field outcomes differ from the paired trajectories")
    labels = {
        "reading": "Reading",
        "arithmetic": "Arithmetic",
        "database": "SQLite",
        "package": "Packages",
        "file": "Files",
        "all": "Pooled",
    }
    header = [
        "family",
        "side",
        "n",
        "correct",
        "completed",
        "tokens",
        "wall_s",
        "first_token_s",
        "peak_bytes",
    ]
    rows = []
    for group in labels:
        for side in data:
            cell = decision["by_family"][group][side]
            rows.append(
                [
                    group,
                    side,
                    *[
                        cell[k]
                        for k in (
                            "n",
                            "correct",
                            "completed",
                            "tokens",
                            "seconds",
                            "first_token_seconds",
                            "peak",
                        )
                    ],
                ]
            )
    with (paper / "mixed-deployment.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    with (paper / "mixed-family-sensitivity.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "omitted_family",
                "requests",
                "candidate_correct",
                "base_correct",
                "candidate_only",
                "base_only",
                "paired_p",
                "candidate_wall_s",
                "base_wall_s",
            ]
        )
        for excluded in labels:
            if excluded == "all":
                continue
            pairs = [
                (a, b)
                for a, b in zip(data["candidate"], data["control"], strict=True)
                if a["family"] != excluded
            ]
            assert all(
                a["id"] == b["id"] and a["family"] == b["family"] for a, b in pairs
            )
            outcomes = [
                (a["score"]["success"], b["score"]["success"]) for a, b in pairs
            ]
            wins = outcomes.count((True, False))
            losses = outcomes.count((False, True))
            writer.writerow(
                [
                    excluded,
                    len(pairs),
                    sum(a for a, _ in outcomes),
                    sum(b for _, b in outcomes),
                    wins,
                    losses,
                    exact_mcnemar(wins, losses),
                    sum(a["wall_s"] for a, _ in pairs),
                    sum(b["wall_s"] for _, b in pairs),
                ]
            )
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.65), sharey=True)
    items = list(labels)
    for i, key in enumerate(items):
        pooled = decision["by_family"][key]

        def values(s):
            return (
                100 * s["correct"] / s["n"],
                s["seconds"] / s["n"],
                1000 * s["first_token_seconds"] / s["n"],
                s["peak"] / 2**30,
            )

        candidate, control = map(values, (pooled["candidate"], pooled["control"]))
        for j, (ax, a, b) in enumerate(zip(axes, candidate, control, strict=True)):
            ax.plot([a, b], [i, i], lw=1, color="#aeb8b4")
            ax.scatter(
                a, i - 0.045, s=22, c="#11675d", label="Recurrent" if not i else None
            )
            ax.scatter(
                b,
                i + 0.045,
                s=20,
                c="#9d5932",
                marker="s",
                label="Base" if not i else None,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0].set_yticks(range(len(items)), [labels[k] for k in items], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-4, 104)
    for ax, title in zip(
        axes,
        ("Correct (%)", "Seconds / query", "Mean TTFT (ms)", "Peak (GiB)"),
        strict=True,
    ):
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[1].legend(
        loc="lower left", bbox_to_anchor=(-0.3, 1.16), frameon=False, ncol=2, fontsize=8
    )
    finish(fig, "mixed-deployment")


def transport(paper, results, result, finish):
    name = "transport-lossless-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    sizes = read(root / "sizes.json")
    if sum(sizes.values()) != d["raw_bytes"] or read(root / "original.json") != read(
        root / "restored.json"
    ):
        raise ValueError("Transport figure cannot verify the installed payload")
    stats = (
        ("Base text weights", d["raw_base_weight_bytes"]),
        ("Installed assistant", d["raw_bytes"]),
        ("Ordinary ZIP", d["ordinary_zip_bytes"]),
        ("Paired encoding", d["bytes"]),
    )
    times = (
        ("Ordinary ZIP pack", d["ordinary_zip_seconds"]),
        ("Paired pack", d["pack_seconds"]),
        ("Paired restore", d["restore_seconds"]),
    )
    with (paper / "transport.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["measure", "unit", "value"])
        w.writerows((label, "bytes", value) for label, value in stats)
        w.writerows((label, "seconds", value) for label, value in times)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.2))
    for ax, cells, unit, scale in zip(
        axes, (stats, times), ("GiB", "Seconds"), (2**30, 1), strict=True
    ):
        for i, (label, value) in enumerate(cells):
            v = value / scale
            ax.barh(
                i, v, color="#11675d" if "aired" in label else "#8a9b98", height=0.53
            )
            ax.text(
                v,
                i,
                f"  {v:.2f}" if unit == "GiB" else f"  {v:.1f}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(range(len(cells)), [label for label, _ in cells], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(v for _, v in cells) / scale * 1.27)
        ax.set_xlabel(unit)
        ax.spines[["right", "top"]].set_visible(False)
    finish(fig, "transport")


def providers(paper, results, result):
    from scripts.field_cli_replay import correct, identical

    name = "tool-language-release-02"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    result(name)
    records = unpack(root / "field-cli.json.gz")
    with (paper / "field-cli.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "provider",
                "request",
                "expected",
                "visible_answer",
                "calls",
                "observations",
                "strict_success",
                "identical",
                "installed_seconds",
                "bundled_seconds",
            ]
        )
        for row in records:
            a, b = row["installed"], row["bundled"]
            if row["correct"] != correct(row) or row["identical"] != identical(a, b):
                raise ValueError(
                    "Bundled provider replay differs from the recorded trajectories"
                )
            calls = [r for r in b["turns"] if "observation" in r]
            writer.writerow(
                [
                    row["kind"],
                    row["prompt"],
                    row["expected"],
                    b["answer"],
                    json.dumps([r["call"] for r in calls], sort_keys=True),
                    json.dumps([r["observation"] for r in calls], sort_keys=True),
                    row["correct"],
                    row["identical"],
                    a["seconds"],
                    b["seconds"],
                ]
            )


def packed_delivery(paper, results, result):
    studies = ["transport-lossless-01", "transport-lossless-02"]
    with (paper / "release-transport.csv").open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "study",
                "payload_files",
                "installed_bytes",
                "ordinary_zip_bytes",
                "lossless_bytes",
                "fraction_smaller_than_zip",
                "ordinary_zip_seconds",
                "encode_seconds",
                "restore_seconds",
                "all_files_identical",
                "offline_tool_parity",
                "archive_sha256",
            ]
        )
        for name in studies:
            root = results / name
            if not (root / "checksums.json").exists():
                continue
            d = result(name)
            inventory = read(root / "original.json")
            assert inventory == read(root / "restored.json") and d["identical"]
            w.writerow(
                [
                    name,
                    len(inventory),
                    d["raw_bytes"],
                    d["ordinary_zip_bytes"],
                    d["bytes"],
                    1 - d["bytes"] / d["ordinary_zip_bytes"],
                    d["ordinary_zip_seconds"],
                    d["pack_seconds"],
                    d["restore_seconds"],
                    d["identical"],
                    d["offline_parity"],
                    d["sha256"],
                ]
            )


def tokenizer_streams(paper, results, result, finish):
    name = "session-tokenizer-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    spec = read(root / "registration.json")
    rep = {"stock": 0, "reuse": 0}
    with (paper / "tokenizer-streams.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "execution_order",
                "policy",
                "repeat",
                "exact_trajectories",
                "tokenizer_constructions",
                "switch_first_token_s",
                "all_first_token_s",
                "total_s",
                "peak_mlx_bytes",
                "peak_process_rss_bytes",
            ]
        )
        for i, side in enumerate(spec["order"]):
            j = rep[side]
            r = d["streams"][side][j]
            rep[side] += 1
            w.writerow(
                [
                    i + 1,
                    side,
                    j + 1,
                    d["exact"][i],
                    r["constructions"],
                    r["switch_ttft"],
                    r["all_ttft"],
                    r["seconds"],
                    r["peak"],
                    r["rss"],
                ]
            )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    for ax, key, title in (
        (axes[0], "switch_ttft", "First-token time on 15 switches (s)"),
        (axes[1], "seconds", "Time for all 50 requests (s)"),
    ):
        for i, (side, color, label) in enumerate(
            (("stock", "#9d5932", "Reload"), ("reuse", "#11675d", "Reuse"))
        ):
            rs = d["streams"][side]
            for j, r in enumerate(rs):
                ax.scatter(
                    i + (j - 0.5) * 0.09, r[key], c=color, marker=("o", "s")[j], s=35
                )
        ax.set_xticks([0, 1], ["Reload", "Reuse"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(title)
    finish(fig, "tokenizer-streams")
