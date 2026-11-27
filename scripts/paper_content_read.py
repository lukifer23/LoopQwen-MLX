"""Publish source selection, long-input measurements and any sealed content-read fit."""

from pathlib import Path

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from scripts.assistance_deliberation import write_csv
from scripts.register_content_read import FIT, PRIMARY, RESULT, DIR, PLAN
from scripts.table_policy_reading import committed

P = Path("paper")
DOC = Path("docs/measurements")


def collect():
    raw = DOC / "full-document-features.json"
    cold = DOC / "long-training-memory.json"
    src = DOC / "content-read-sources.json"
    for p in (raw, cold, src, PLAN):
        committed(p)
    a = read(raw)
    b = read(cold)
    figures = []
    for r in a["rows"]:
        first, last = r["source"]
        figures.append(
            dict(
                package=r["key"],
                document_tokens=a["prefix_tokens"],
                start=first,
                stop=last,
                question_tokens=r["opening"] - a["prefix_tokens"],
                target_positions=r["positions"],
                branch_seconds=r["seconds"],
            )
        )
    write_csv(P / "full-document-branches.csv", figures)
    fig, ax = plt.subplots(figsize=(7.1, 2.8), layout="constrained")
    ax.scatter(
        [f["start"] / 1000 for f in figures],
        [f["branch_seconds"] for f in figures],
        color="#34666c",
        s=28,
    )
    ax.axvline(131.072, ls="--", lw=1, color="#9a4a3b")
    ax.annotate(
        "131,072 tokens",
        xy=(131.072, 0.94),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set(
        xlabel="Start of the source version in the original opening (thousands of tokens)",
        ylabel="Time per branch (s)",
        xlim=(0, 151),
        ylim=(0, None),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "full-document-branches.svg", metadata={"Date": None})
    plt.close(fig)
    out = dict(
        full=a,
        parity=b,
        source=read(src),
        inventory=read(DIR / "inventory.json"),
        plan=digest(PLAN),
    )
    result = RESULT / FIT
    if (result / "checksums.json").exists():
        committed(result / "checksums.json")
        out["fit"] = read(result / "decision.json")
        out["jobs"] = [
            job
            for session in sorted(Path("research/sessions").glob("window-*.json"))
            for job in read(session)["jobs"]
            if job["job"] == FIT
        ]
        if (result / "training-log.json.gz").is_file():
            hist = unpack(result / "training-log.json.gz")
            write_csv(
                P / "content-read-updates.csv",
                [
                    dict(
                        update=i + 1,
                        **{
                            k: r[k]
                            for k in (
                                "epoch",
                                "positions",
                                "marks",
                                "loss",
                                "nll",
                                "gradient",
                                "evidence",
                                "kl",
                            )
                        },
                    )
                    for i, r in enumerate(hist)
                ],
            )
    replay = DOC / (FIT + "-cold.json")
    if replay.exists():
        committed(replay)
        out["replay"] = read(replay)
    reading = DOC / "content-read-reading.json"
    if reading.exists():
        from scripts.paper_learner_prefix import paired_requests

        root = RESULT / PRIMARY
        for p in (reading, root / "checksums.json"):
            committed(p)
        out["primary"] = read(root / "decision.json")
        out["reading"] = read(reading)
        paired_requests(root, "content-read")
    trials = {}
    for prefix in ("content-read", "bounded-read"):
        name = prefix + "-fit-01"
        result = RESULT / name
        if not (result / "checksums.json").is_file():
            continue
        committed(result / "checksums.json")
        fit = read(result / "decision.json")
        complete = (result / "training-log.json.gz").is_file()
        hist = unpack(
            result
            / ("training-log.json.gz" if complete else "partial-training-log.json.gz")
        )
        trials[prefix] = (fit, hist)
        if prefix == "bounded-read":
            out["bounded"] = fit
            out["bounded_jobs"] = [
                job
                for session in Path("research/sessions").glob("window-*.json")
                for job in read(session)["jobs"]
                if job["job"] == name
            ]
            replay = DOC / (name + "-cold.json")
            if replay.is_file():
                committed(replay)
                out["bounded_cold"] = read(replay)
            write_csv(
                P / "bounded-read-updates.csv",
                [
                    dict(
                        update=i + 1,
                        **{
                            k: r[k]
                            for k in (
                                "epoch",
                                "positions",
                                "marks",
                                "loss",
                                "nll",
                                "gradient",
                                "evidence",
                                "kl",
                            )
                        },
                    )
                    for i, r in enumerate(hist)
                ],
            )
        out[prefix + "_early"] = [w["kl"] for w in hist[:16]]
    if trials:
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3), layout="constrained")
        colors = ["#af6246", "#2e6776"]
        groups = [
            "reasoning",
            "calculate",
            "quantity",
            "reuse",
            "redirect",
            "instruction",
        ]
        for i, (prefix, (fit, hist)) in enumerate(trials.items()):
            label = "Unbounded write" if i == 0 else "Bounded write"
            axes[0].plot(
                range(1, len(hist[:16]) + 1),
                [w["kl"] for w in hist[:16]],
                marker=".",
                ms=4,
                color=colors[i],
                label=label,
            )
            if "gains" in fit:
                axes[1].barh(
                    [j + (i - 0.5) * 0.35 for j in range(len(groups))],
                    [fit["gains"][g] for g in groups],
                    height=0.33,
                    color=colors[i],
                    label=label,
                )
        axes[0].set(
            xlabel="Native optimizer batch", ylabel="KL from frozen parent (nats/token)"
        )
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].axvline(0, color="#666666", linewidth=0.7)
        axes[1].set_yticks(
            range(len(groups)),
            [
                "Reasoning",
                "Calculator call",
                "Quantity and unit",
                "Use prior SQL",
                "Redirect SQL",
                "Instructions",
            ],
        )
        axes[1].set_xlabel("Gain in target NLL (nats/token)")
        axes[1].invert_yaxis()
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(P / "content-write-optimization.svg", metadata={"Date": None})
        plt.close(fig)
    bounded_reading = DOC / "bounded-read-reading.json"
    if bounded_reading.is_file():
        from scripts.paper_learner_prefix import paired_requests

        root = RESULT / "bounded-read-transfer-01"
        for p in (bounded_reading, root / "checksums.json"):
            committed(p)
        out["bounded_primary"] = read(root / "decision.json")
        out["bounded_reading"] = read(bounded_reading)
        paired_requests(root, "bounded-read")
    for name in (
        "full-document-adjoint",
        "document-generation-parity",
        "full-document-kernel",
    ):
        p = DOC / (name + ".json")
        if p.exists():
            committed(p)
            out[name.replace("-", "_")] = read(p)
    delivery = {}
    for stage in (
        "staged-query-transfer-01",
        "learner-prefix-transfer-01",
        "bounded-read-transfer-01",
    ):
        p = DOC / (stage + "-joint-delivery.json")
        if p.is_file():
            committed(p)
            receipt = read(p)
            delivery[stage] = dict(
                weights=receipt["source"]["policy"],
                all=next(r for r in receipt["groups"] if r["family"] == "all"),
            )
    out["delivery"] = delivery
    return out
