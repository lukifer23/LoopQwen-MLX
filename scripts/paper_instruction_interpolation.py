"""Render the paired outcomes and all complete responses from the immutable field archive."""

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader

from loopqwen.io import digest, read
from loopqwen.research.native_function_study import executions

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NAME = "instruction-interpolation"
STUDY = NAME + "-01"


def main():
    trial = ROOT / "research/results" / STUDY
    checks = read(trial / "checksums.json")
    registry = read(ROOT / "research/registry.json")
    assert digest(trial / "checksums.json") == registry["experiments"][STUDY]["archive"]
    assert all(digest(trial / k) == v for k, v in checks.items())
    spec = read(trial / "registration.json")
    assert (
        digest(ROOT / spec["material"])
        == registry["datasets"][spec["material"]]["sha256"]
    )
    panel = read(ROOT / spec["material"])
    dec = read(trial / "decision.json")
    sides = ("candidate", "control")
    runs = {
        s: json.loads(gzip.decompress((trial / (s + ".json.gz")).read_bytes()))
        for s in sides
    }
    rows = []
    destinations = defaultdict(Counter)
    for i, w in enumerate(panel["rows"]):
        for j, question in enumerate(w["questions"]):
            row = {
                "id": w["id"],
                "unit": w["cluster"],
                "family": w["family"],
                "message": j + 1,
                "question": question,
                "gold": w["gold"] if w["family"] == "reference" else w["gold"][j],
                "source": w.get("author", w.get("document")),
            }
            for s in sides:
                r = runs[s][i]
                assert w["id"] == r["id"]
                q = r["requests"][j]
                tr = q["trajectory"]
                events = executions(tr)
                evidence = {
                    "answer": tr["answer"],
                    "status": tr["status"],
                    **q["score"],
                    "calls_and_returns": json.dumps(events, ensure_ascii=False),
                    "opening_tokens": len(tr["turns"][0]["prompt_ids"]),
                    "output_tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                    "first_s": q["first_s"],
                    "wall_s": q["wall_s"],
                    "peak_bytes": q["peak"],
                }
                if w["family"] == "inventory":
                    first = next(
                        (x for x in events if x["call"]["name"] == "csv_lookup"), None
                    )
                    field = (
                        first["call"]["arguments"]["key_column"]
                        if first
                        else "(no lookup)"
                    )
                    destinations[s][field] += 1
                row.update({s + "_" + k: v for k, v in evidence.items()})
            rows.append(row)
    with (PAPER / (NAME + "-requests.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)
    with (PAPER / (NAME + "-costs.csv")).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Measurement", "Loop", "Base"])
        for label, key, scale in [
            ("Cold stream wall (s)", "wall_s", 1),
            ("Sum of first-token waits (s)", "first_s", 1),
            ("Maximum MLX allocation (GiB)", "peak", 1 / 2**30),
            ("Opening tokens / first seconds", "opening", 1),
            ("Post-first tokens / second", "decode", 1),
            ("Total output tokens", "output_tokens", 1),
        ]:
            vals = [
                (
                    dec["groups"][s]["rates"]
                    if key in ("opening", "decode")
                    else dec["groups"][s]["all"]
                )[key]
                * scale
                for s in sides
            ]
            writer.writerow(
                [
                    label,
                    *[
                        f"{v:,.2f}" if key != "output_tokens" else f"{v:,}"
                        for v in vals
                    ],
                ]
            )
    metrics = [
        (
            "Meaning requests",
            40,
            [dec["groups"][s]["reference"]["answers"] for s in sides],
        ),
        (
            "Both meanings",
            20,
            [dec["pairs"]["reference"][arm] for arm in ("loop", "base")],
        ),
        (
            "Grounded first lookup",
            8,
            [
                sum(
                    r[s + "_success"]
                    for r in rows
                    if r["family"] == "inventory" and r["message"] == 1
                )
                for s in sides
            ],
        ),
        (
            "Grounded difference",
            8,
            [
                sum(
                    r[s + "_success"]
                    for r in rows
                    if r["family"] == "inventory" and r["message"] == 2
                )
                for s in sides
            ],
        ),
        (
            "Both file requests",
            8,
            [dec["pairs"]["inventory"][arm] for arm in ("loop", "base")],
        ),
    ]
    with (PAPER / (NAME + "-accuracy.csv")).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Endpoint", "N", "Loop", "Base"])
        writer.writerows([[k, n, *v] for k, n, v in metrics])
    matplotlib.rcParams.update(
        {"svg.hashsalt": NAME, "font.family": "DejaVu Sans", "font.size": 9}
    )
    fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
    for i, (arm, color, offset) in enumerate(
        [("Loop", "#146456", -0.18), ("Base", "#956035", 0.18)]
    ):
        yy = [j + offset for j in range(len(metrics))]
        values = [100 * v[i] / n for _, n, v in metrics]
        ax.barh(yy, values, height=0.31, color=color, label=arm)
        for y, x, (_, n, v) in zip(yy, values, metrics, strict=True):
            ax.text(x + 1.2, y, f"{v[i]}/{n}", va="center", fontsize=8)
    ax.set_yticks(range(len(metrics)), [m[0] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of requests or complete pairs solved (%)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(PAPER / (NAME + ".svg"), metadata={"Date": None})
    plt.close(fig)
    target = PAPER / (NAME + ".pdf")
    typst.compile(str(PAPER / (NAME + ".typ")), output=str(target), timestamp=0)
    pages = PdfReader(target).pages
    assert all(len(page.extract_text()) > 240 for page in pages)
    print(len(pages), "pages", len(rows), "paired requests", dict(destinations))


if __name__ == "__main__":
    main()
