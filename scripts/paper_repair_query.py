"""Render the two-read source-repair study from committed qualification, fit and paired outputs."""

from collections import Counter
from pathlib import Path

import matplotlib
import typst
from pypdf import PdfReader, PdfWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv
from scripts.register_repair_query import (
    DOC,
    FEATURES,
    FIT,
    PANEL,
    PRIMARY,
    RESULT,
    ROOT,
)
from scripts.table_policy_reading import committed

P = Path("paper")


def category(group):
    if group == "strategy":
        return "Human multi-step reasoning"
    if group.startswith("report_"):
        return "Expert financial reports"
    if group in (
        "inspect_after_error",
        "repair_after_schema",
        "repair_known_schema",
        "deliver_corrected_result",
    ):
        return "Database error repair"
    if group in (
        "repair_csv_exact",
        "repair_via_sql",
        "deliver_csv",
        "deliver_sql",
        "repair_column_roles",
        "answer_after_role_repair",
    ):
        return "CSV key and column repair"
    if group in ("answer_valid_read", "redirect_valid_read", "answer_after_redirect"):
        return "Misleading successful SQL"
    return "Earlier assistance"


def stage(name):
    root = RESULT / name
    if not (root / "checksums.json").exists():
        return None
    committed(root / "checksums.json")
    if name == PRIMARY and (root / "control.json.gz").is_file():
        committed(DOC / "repair-query-reading.json")
    jobs = [
        r
        for path in Path("research/sessions").glob("window-*.json")
        for r in read(path)["jobs"]
        if r["job"] == name
    ]
    s = read(root / "registration.json")
    d = read(root / "decision.json")
    row = {
        "stage": name,
        "decision": d,
        "seconds": sum(r["seconds"] for r in jobs),
        "peak": max((r["peak_bytes"] / 2**30 for r in jobs), default=0),
        "budget": s["limits"]["wall_seconds"],
        "sha256": digest(root / "checksums.json"),
    }
    path = DOC / (name + "-cold.json")
    if path.exists():
        committed(path)
        row["cold"] = {k: v for k, v in read(path).items() if k != "checks"}
    return row


def chart(rows):
    plt.rcParams.update({"svg.hashsalt": "repair-query-evidence", "font.size": 8.5})
    keys = list(rows)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(9.5, 2.6), sharey=True, layout="constrained"
    )
    for ax, k, label, color in (
        (a, "opening_positions", "Input tokens (millions)", "#306f77"),
        (b, "target_positions", "Supervised target tokens (thousands)", "#ac643e"),
    ):
        div = 1e6 if ax is a else 1e3
        ax.barh(keys, [rows[v][k] / div for v in keys], color=color)
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "repair-query-exposure.svg", metadata={"Date": None})
    plt.close(fig)


def collect():
    stages = [r for n in (FEATURES, FIT, PRIMARY) if (r := stage(n))]
    if not stages:
        raise ValueError("At least one closed native stage is required for this paper")
    inv = read(ROOT / "verified-inventory.json")
    groups = {}
    for g, counts in inv["groups"].items():
        groups.setdefault(category(g), Counter()).update(counts)
    chart(groups)
    write_csv(
        P / "repair-query-exposure.csv",
        [{"category": k, **v} for k, v in groups.items()],
    )
    out = {
        "stages": stages,
        "groups": groups,
        "training": inv["training"],
        "examples": inv["examples"],
        "positions": inv["target_positions"],
        "citations": inv["human_cited_queries"],
        "opening": inv["opening_positions"],
        "stages_done": len(stages),
        "panel": digest(PANEL),
        "panel_check": digest(ROOT / "panel-replay.json"),
        "precision": digest(DOC / "shared-query-precision-idle.json"),
    }
    fitroot = RESULT / FIT
    if len(stages) >= 2 and (fitroot / "training-log.json.gz").exists():
        history = unpack(fitroot / "training-log.json.gz")
        write_csv(
            P / "repair-query-updates.csv",
            [{k: v for k, v in r.items() if k != "rows"} for r in history],
        )
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.35), layout="constrained")
        for ax, (metric, divisor, label) in zip(
            axes,
            (
                ("ce", "positions", "Response NLL per token"),
                ("kl", "positions", "Teacher KL per token"),
                ("citation_nll", "citations", "NLL per evidence query"),
            ),
            strict=True,
        ):
            samples = [history[i : i + 16] for i in range(0, len(history), 16)]
            values = [
                (
                    h[-1]["update"],
                    sum(r[metric] for r in h) / sum(r[divisor] for r in h),
                )
                for h in samples
                if sum(r[divisor] for r in h)
            ]
            ax.plot(
                [i for i, _ in values], [z for _, z in values], color="#306f77", lw=1
            )
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Completed optimizer update")
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(P / "repair-query-optimization.svg", metadata={"Date": None})
        plt.close(fig)
        out["optimization"] = True
    primary = RESULT / PRIMARY
    if len(stages) == 3 and (primary / "control.json.gz").exists():
        panel = unpack(PANEL)["rows"]
        runs = [
            unpack(primary / (name + ".json.gz")) for name in ("candidate", "control")
        ]
        rows = []
        for w, c, b in zip(panel, *runs, strict=True):
            assert w["id"] == c["id"] == b["id"]
            row = {
                "id": w["id"],
                "family": w["family"],
                "source": w["cluster"],
                "condition": w.get("condition", ""),
                "opening": len(w["initial_ids"]),
            }
            for a, v in (("loop", c), ("base", b)):
                q = v["requests"][0]
                assert q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                row.update(
                    {
                        a + "_" + k: x
                        for k, x in {
                            "success": q["score"]["success"],
                            "completed": q["score"]["completed"],
                            "wall_s": q["wall_s"],
                            "first_s": q["first_s"],
                            "peak_bytes": q["peak"],
                            "output_tokens": sum(
                                len(t["token_ids"]) for t in q["trajectory"]["turns"]
                            ),
                            "visible": q["trajectory"]["answer"],
                        }.items()
                    }
                )
            rows.append(row)
        write_csv(P / "repair-query-requests.csv", rows)
        out["reading"] = read(DOC / "repair-query-reading.json")
    return out


def build():
    out = collect()
    save(P / "repair-query.json", out)
    pdf = P / "repair-query.pdf"
    typst.compile(str(P / "repair-query.typ"), output=str(pdf), root=".", timestamp=0)
    doc = PdfReader(pdf)
    assert all(len(p.extract_text()) > 130 for p in doc.pages)
    prior = PdfReader(P / "loopqwen-provider-recovery-study.pdf")
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Title": "Can a recurrent read of source memory repair a plausible wrong answer?",
            "/Author": "LoopQwen-MLX",
        }
    )
    writer.append(
        doc,
        outline_item="Shared query recurrence, trained source repair and measured prerequisites",
    )
    writer.append(prior, outline_item="Prior methods, training and direct evaluations")
    dest = P / "loopqwen-repair-query-study.pdf"
    writer.write(dest)
    assert len(PdfReader(dest).pages) == len(doc.pages) + len(prior.pages)
    print(
        len(doc.pages),
        "local pages;",
        len(PdfReader(dest).pages),
        "total pages;",
        digest(dest),
    )


if __name__ == "__main__":
    build()
