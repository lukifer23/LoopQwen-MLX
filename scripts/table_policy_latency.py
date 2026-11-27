"""Explain measured latency using matched initial tokens and all real generated turns."""

import itertools
import math
from pathlib import Path
from statistics import median

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, save
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_reading import ROOT

P = Path("paper")
ARMS = ("candidate", "control")
BOUNDS = (128, 256, 512, 768, math.inf)


def observations(arm):
    records = unpack(ROOT / f"{arm}.json.gz")
    for record in records:
        for number, request in enumerate(record["requests"]):
            turns = request["trajectory"]["turns"]
            first = turns[0]
            assert len(first["prompt_ids"]) > 0 and first["reused_prompt_tokens"] == 0
            yield (
                (record["id"], number),
                first["prompt_ids"],
                {
                    "first": request["first_s"],
                    "opening": first["ttft_s"],
                    "other_prefill": sum(t["ttft_s"] for t in turns[1:]),
                    "decode": sum(t["seconds"] - t["ttft_s"] for t in turns),
                    "tokens": sum(len(t["token_ids"]) - 1 for t in turns),
                    "turns": len(turns),
                    "wall": request["wall_s"],
                    "outside": request["wall_s"] - sum(t["seconds"] for t in turns),
                },
            )


def build():
    paired = []
    for a, b in zip(*(observations(arm) for arm in ARMS), strict=True):
        assert a[:2] == b[:2]
        n = len(a[1])
        bound = next(k for k in BOUNDS if n <= k)
        paired.append((bound, n, a[2], b[2]))
    assert len(paired) == 268
    bins = []
    for bound, values in itertools.groupby(
        sorted(paired, key=lambda r: r[0]), lambda r: r[0]
    ):
        bucket = list(values)
        row = {
            "length": {
                128: "1–128",
                256: "129–256",
                512: "257–512",
                768: "513–768",
                math.inf: "769+",
            }[bound],
            "requests": len(bucket),
            "initial_tokens": sum(r[1] for r in bucket),
        }
        for i, arm in enumerate(ARMS, 2):
            opening = sum(r[i]["opening"] for r in bucket)
            row[f"{arm}_opening_tps"] = row["initial_tokens"] / opening
            row[f"{arm}_opening_s"] = opening
            row[f"{arm}_median_first_ms"] = median(r[i]["first"] * 1000 for r in bucket)
        row["loop_faster_first"] = sum(r[2]["first"] < r[3]["first"] for r in bucket)
        bins.append(row)
    arms = {
        arm: {key: sum(r[i][key] for r in paired) for key in paired[0][i]}
        for i, arm in enumerate(ARMS, 2)
    }
    c, b = (arms[k] for k in ARMS)
    rates = [r["decode"] / r["tokens"] for r in (c, b)]
    components = {
        "Generated token count": (c["tokens"] - b["tokens"]) * sum(rates) / 2,
        "Seconds per decoded token": (rates[0] - rates[1])
        * (c["tokens"] + b["tokens"])
        / 2,
        "Identical first openings": c["opening"] - b["opening"],
        "Later tool-turn openings": c["other_prefill"] - b["other_prefill"],
        "Outside timed generation": c["outside"] - b["outside"],
    }
    assert math.isclose(sum(components.values()), c["wall"] - b["wall"], abs_tol=1e-8)
    write_csv(P / "table-policy-opening-length.csv", bins)
    audit = {
        "primary": digest(ROOT / "checksums.json"),
        "bins": bins,
        "arms": arms,
        "components": components,
    }
    save(P / "table-policy-latency.json", audit)
    return audit


def render(audit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import typst

    plt.rcParams.update(
        {
            "svg.hashsalt": "four-layer-openings",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 2.75), layout="constrained")
    bins = audit["bins"]
    for arm, label, color in zip(
        ARMS, ("Trained loop", "Untouched Base"), ("#006c61", "#9e4f25"), strict=True
    ):
        left.plot(
            [r["length"] for r in bins],
            [r[f"{arm}_median_first_ms"] for r in bins],
            marker="o",
            label=label,
            color=color,
        )
    left.set(
        xlabel="Tokens in identical initial prompt", ylabel="Median first token (ms)"
    )
    left.tick_params(axis="x", labelrotation=27)
    left.legend(frameon=False)
    parts = audit["components"]
    names = (
        "Output length",
        "Decode rate",
        "First opening",
        "Later openings",
        "Other time",
    )
    values = list(parts.values())
    right.barh(names, values, color=["#006c61" if v < 0 else "#9e4f25" for v in values])
    right.axvline(0, lw=0.65, color="#333333")
    right.set(xlabel="Loop minus Base time (seconds)")
    right.invert_yaxis()
    fig.savefig(P / "table-policy-latency.svg", metadata={"Date": None})
    plt.close(fig)
    pdf = typst.compile(str(P / "table-policy-latency.typ"), timestamp=0)
    (P / "table-policy-latency.pdf").write_bytes(pdf)
    return pdf


if __name__ == "__main__":
    render(build())
