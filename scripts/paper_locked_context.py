"""Publish the two full lockfile comparisons from their original installed CLI streams."""

import csv
import json
from pathlib import Path

import matplotlib
import typst
from pypdf import PdfReader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.io import read, save
from loopqwen.evaluation.mechanism import unpack
from loopqwen.research.cross_view_study import tokenizer
from scripts.assemble_evidence_paper import edition
from scripts.locked_context_material import MATERIAL, SOURCE
from scripts.locked_context_study import NAME, verify

P = Path("paper")
R = Path("research/results") / NAME


def run():
    d = read(R / "decision.json")
    verify(R, d)
    if "rows" not in d:
        print(
            "A resource-limited run did not produce a complete four-request comparison."
        )
        return
    source = {r["revision"]: r for r in read(MATERIAL)["rows"]}
    loc = {
        r["revision"]: r["version_start_tokens"]
        for r in read(SOURCE / "answer-locations.json")
    }
    tok = tokenizer(read(R / "registration.json"))
    raw_pairs = unpack(R / "requests.json.gz")
    raw = {
        (ref, side): pair["runs"][side][0]
        for ref, pair in zip(source, raw_pairs, strict=True)
        for side in ("recurrent", "base")
    }
    records = []
    for r in d["rows"]:
        w = source[r["revision"]]
        turns = raw[r["revision"], r["side"]]["turns"]
        generated = [tok.decode(t["token_ids"]) for t in turns]
        thoughts = [t.partition("</think>")[0] for t in generated]
        records.append(
            {
                **r,
                "file_bytes": w["bytes"],
                "thinking_contains_version": any(w["version"] in t for t in thoughts),
                "thinking_boundaries": sum("</think>" in t for t in generated),
                "pinned_version_repetitions": sum(
                    t.count("I need its pinned version.") for t in thoughts
                ),
                "byte_count_repetitions": sum(
                    t.count("I need its byte count.") for t in thoughts
                ),
                "pinned_version": w["version"],
                "version_offsets": json.dumps(loc[r["revision"]]),
                "calls": json.dumps(
                    r["calls"], ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    with (P / "locked-context-requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update(
        {
            "svg.hashsalt": "locked-context-01",
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained")
    for j, (ref, w) in enumerate(source.items()):
        a.hlines(j, 0, w["prompt_tokens"], color="#b7c9c8", linewidth=6)
        a.plot(
            131072,
            j,
            "|",
            color="#374c4b",
            markersize=14,
            label="131,072 tokens" if j == 0 else None,
        )
        a.plot(
            loc[ref][0],
            j,
            "D",
            color="#20736a",
            markersize=6,
            label="First version occurrence" if j == 0 else None,
        )
        for idx, (side, label, c) in enumerate(
            (
                ("recurrent", "Trained loop", "#16695f"),
                ("base", "Untouched Base", "#6a619a"),
            )
        ):
            r = next(r for r in records if r["revision"] == ref and r["side"] == side)
            b.bar(
                j + (idx - 0.5) * 0.34,
                r["wall_s"],
                0.31,
                color=c,
                label=label if j == 0 else None,
            )
    a.set(
        yticks=[0, 1],
        yticklabels=["Earlier", "Later"],
        ylim=(-0.6, 1.6),
        xlim=(0, 155000),
        xlabel="Position in whole input",
    )
    a.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    b.set(xticks=[0, 1], xticklabels=["Earlier", "Later"], ylabel="End-to-end wait (s)")
    for ax in (a, b):
        ax.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=8
        )
    fig.savefig(P / "locked-context.svg", metadata={"Date": None})
    plt.close(fig)
    totals = d["totals"]
    table = []
    for lab, k, scale, precision in (
        ("Correct version", "version", 1, 0),
        ("Correct byte count", "size", 1, 0),
        ("Executed size check", "measured", 1, 0),
        ("Completed responses", "complete", 1, 0),
        ("Both correct and grounded", "grounded", 1, 0),
        ("Total user wait (s)", "wall_s", 1, 2),
        ("Summed first-token waits (s)", "first_s", 1, 2),
        ("Peak MLX allocation (GiB)", "peak", 2**30, 3),
        ("Initial prompt rate (tokens/s)", "prefill_tps", 1, 2),
        ("Decode rate (tokens/s)", "decode_tps", 1, 2),
        ("Generated tokens", "generated", 1, 0),
        ("Reused initial tokens", "reuse", 1, 0),
    ):
        table.append(
            [
                lab,
                *[
                    f"{totals[side][k] / scale:,.{precision}f}"
                    for side in ("recurrent", "base")
                ],
            ]
        )
    examples = []
    for r in records:
        examples.append(
            [
                "Earlier" if r["revision"] == next(iter(source)) else "Later",
                "Loop" if r["side"] == "recurrent" else "Base",
                str(len(json.loads(r["calls"]))),
                r["status"],
                (r["answer"] or "No completed answer")[:240],
            ]
        )
    missing = [k for k, v in d["gates"].items() if not v]
    save(
        P / "locked-context-paper.json",
        {
            "diagnostic": f"The recurrent model generated 256 tokens on each file without reaching a thinking boundary. It repeated the same pinned-version sentence {records[0]['pinned_version_repetitions']} and {records[2]['pinned_version_repetitions']} times. Base called file_lines on the first revision and received 1,386 lines. On the later file its unfinished deliberation contained the correct version, 2026.11.2. Neither system delivered a version or byte count in a completed response.",
            "metrics": table,
            "responses": examples,
            "missing": ", ".join(missing) if missing else "none",
            "summary": f"Release 05 gave {totals['recurrent']['grounded']}/2 grounded answers and untouched Base gave {totals['base']['grounded']}/2. "
            + (
                "All registered criteria passed."
                if d["passed"]
                else "The complete registered gate was missed."
            ),
        },
    )
    raw = typst.compile(str(P / "locked-context.typ"), timestamp=0)
    (P / "locked-context.pdf").write_bytes(raw)
    doc = PdfReader(P / "locked-context.pdf")
    assert len(doc.pages) == 2 and all(len(p.extract_text()) > 400 for p in doc.pages)
    edition(P, "locked-context")
    edition(P, "locked-context", verify=True)


if __name__ == "__main__":
    run()
