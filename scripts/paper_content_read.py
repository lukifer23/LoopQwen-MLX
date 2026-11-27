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
    for name in ("full-document-adjoint", "document-generation-parity"):
        p = DOC / (name + ".json")
        if p.exists():
            committed(p)
            out[name.replace("-", "_")] = read(p)
    return out
