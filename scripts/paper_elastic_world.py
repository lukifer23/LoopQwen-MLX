"""Rebuild the resource continuation and real-provider evaluation supplement from sealed receipts."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader, PdfWriter

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv
from scripts.contrastive_fit_specificity import run as specificity
from scripts.paper_contrastive_world import DOC, P, RESULT, held, optimizer_plot
from scripts.table_policy_reading import committed

ATTEMPTS = ["contrastive-world-fit-01", "streamed-world-fit-01", "elastic-world-fit-01"]
MATERIAL = Path("research/sources/provider-recovery")


def history():
    output = []
    log = []
    for name in ATTEMPTS:
        root = RESULT / name
        if not (root / "checksums.json").exists():
            break
        committed(root / "checksums.json")
        decision = read(root / "decision.json")
        jobs = [
            j
            for path in Path("research/sessions").glob("window-*.json")
            for j in read(path)["jobs"]
            if j["job"] == name
        ]
        assert jobs
        if (root / "partial-state.json").exists():
            rows = read(root / "partial-state.json")["log"]
        elif (root / "training-log.json.gz").exists():
            rows = unpack(root / "training-log.json.gz")
        else:
            rows = log
        assert rows[: len(log)] == log
        log = rows
        output.append(
            {
                "name": name,
                "updates": len(log),
                "seconds": sum(r["seconds"] for r in jobs),
                "peak": max(r["peak_bytes"] for r in jobs) / 2**30,
                "targets": sum(r["targets"] for r in log),
                "preferences": sum(r["preferences"] for r in log),
                "archive": digest(root / "checksums.json"),
                "decision": decision,
            }
        )
    return output, log


def figure(data):
    plt.rcParams.update({"svg.hashsalt": "provider-recovery-field", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 2.3), layout="constrained")
    rows = sorted(data["dev"]["by_database"].items())
    names = [k.replace("_", " ") for k, _ in rows]
    sets = [[r for r in v["distilled"] if not r["named_original"]] for _, v in rows]
    n = [len(s) for s in sets]
    ax.barh(names, n, color="#c27642", label="Partners return same result")
    ax.barh(
        names,
        [sum(z["distinguishes"] for z in s) for s in sets],
        color="#306f77",
        label="Partners return different results",
    )
    for i, v in enumerate(n):
        ax.text(v + 0.5, i, str(v), va="center")
    ax.set_xlim(0, max(n) + 6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Published alternate SQLite instances per development schema")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "provider-field-denotations.svg", metadata={"Date": None})
    plt.close(fig)


def build():
    stages, log = history()
    optimizer_plot(log, prefix="pressure-continuation")
    write_csv(P / "pressure-continuation-optimizer.csv", log)
    fields = read(MATERIAL / "field-denotations.json")["panels"]
    figure(fields)
    names = [
        "qualification",
        "csv-qualification",
        "csv-role-qualification",
        "semantic-qualification",
    ]
    corpora = [read(MATERIAL / (n + ".json")) for n in names]
    wording = read(MATERIAL / "field-wording.json")["items"]
    proofs = [read(RESULT / "streamed-world-fit-01" / "gradient-parity.json")]
    elastic = RESULT / ATTEMPTS[-1]
    if (elastic / "checksums.json").exists() and (
        elastic / "gradient-parity.json"
    ).exists():
        proofs.append(read(elastic / "gradient-parity.json"))
    out = {
        "stages": stages,
        "latest": stages[-1],
        "target": read("research/experiments/elastic-world-fit-01.json")[
            "expected_positions"
        ],
        "gradient_relative": max(r["gradient_relative"] for rr in proofs for r in rr),
        "objective_relative": max(r["objective_relative"] for rr in proofs for r in rr),
        "cosine": min(r["gradient_cosine"] for rr in proofs for r in rr),
        "training": [
            {"turns": s.get("turns", s.get("supervised_turns")), "labels": s["labels"]}
            for s in corpora
        ],
        "fields": {
            k: {
                key: v
                for key, v in d.items()
                if key not in ("by_database", "changed_from_original")
            }
            for k, d in fields.items()
        },
        "flags": sum(r["tag"] != "no_specific_ambiguity_flagged" for r in wording),
        "references": {
            str(p): digest(p)
            for p in [
                MATERIAL / "field-denotations.json",
                MATERIAL / "field-wording.json",
                MATERIAL / "semantic-independent-replay.json",
                DOC / "shared-query-precision.json",
            ]
        },
    }
    if (elastic / "training-free-generation.json.gz").exists():
        out["specificity"] = specificity(True, name=ATTEMPTS[-1])
    primary = RESULT / "elastic-world-transfer-01"
    if (primary / "checksums.json").exists():
        committed(primary / "checksums.json")
        if (primary / "control.json.gz").is_file():
            for group in ("services", "qualitative"):
                committed(DOC / ("elastic-" + group + "-reading.json"))
            held(out, variant="elastic")
        out["held_decision"] = read(primary / "decision.json")
    save(P / "pressure-continuation.json", out)
    pdf = P / "pressure-continuation.pdf"
    typst.compile(
        str(P / "pressure-continuation.typ"), output=str(pdf), root=".", timestamp=0
    )
    local = PdfReader(pdf)
    assert all(len(page.extract_text()) > 170 for page in local.pages)
    prior = P / "loopqwen-contrastive-world-study.pdf"
    combined = P / "loopqwen-provider-recovery-study.pdf"
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Title": "Training recurrence under memory pressure and testing human tool requests",
            "/Author": "LoopQwen-MLX",
        }
    )
    writer.append(
        local, outline_item="Measured continuation and independent provider validation"
    )
    writer.append(
        prior,
        outline_item="Recurrent architecture, masked evaluations and complete historical studies",
    )
    writer.write(combined)
    assert len(PdfReader(combined).pages) == len(local.pages) + len(
        PdfReader(prior).pages
    )
    print(
        f"{len(local.pages)} supplement pages; {len(PdfReader(combined).pages)} total; {digest(combined)}"
    )


if __name__ == "__main__":
    build()
