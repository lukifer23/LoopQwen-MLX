"""Publish actual broad-response fitting, freely decoded programs, and signed preference changes."""

from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader, PdfWriter

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv
from scripts.paper_conditional_features import DOC, GROUPS, P

R = Path("research/results/conditional-program-fit-01")


def build():
    decision, fit, spec = (
        read(R / name) for name in ("decision.json", "fit.json", "registration.json")
    )
    replay = read(DOC / "conditional-trained-native-replay.json")
    assert (
        replay["archive"] == digest(R / "checksums.json")
        and replay["weights"] == fit["weights"]
    )
    steps = unpack(R / "training-log.json.gz")
    probes = read(R / "training-probes.json")
    assert len(steps) == decision["updates"] and len(probes) == 56
    write_csv(P / "conditional-fit-steps.csv", steps)
    write_csv(P / "conditional-fit-probes.csv", replay["records"])
    ablations = {
        r["id"]: r["gain_from_actual_states_over_repeated_ordinary_state"]
        for r in replay["records"]
    }
    groups = []
    for group, label in GROUPS.items():
        g = [r for r in probes if r["group"] == group]
        assert len(g) == 8
        orig, trained = [mean(r[k] for r in g) for k in ("ordinary", "trained")]
        assert abs(orig - trained - decision["group_mean_nll_gain"][group]) < 1e-6
        groups.append(
            {
                "label": label,
                "group": group,
                "before": orig,
                "after": trained,
                "gain": orig - trained,
                "state_information_gain": mean(ablations[r["id"]] for r in g),
                "improved": sum(r["ordinary"] > r["trained"] for r in g),
                "positions": sum(r["positions"] for r in g),
            }
        )
    margin_rows = [
        {
            "id": r["id"],
            "edit": r["edit"],
            "correct_changed_tokens": len(r["chosen_positions"]),
            "incorrect_changed_tokens": len(r["rejected_positions"]),
            **{
                k + "_" + stat: value
                for k in ("ordinary", "trained")
                for stat, value in r[k].items()
            },
        }
        for r in replay["training_sql_span_margins"]
    ]
    write_csv(P / "conditional-sql-margins.csv", margin_rows)
    margins = []
    for op in ("column", "predicate", "direction", "all"):
        rows = [r for r in margin_rows if op == "all" or r["edit"] == op]
        v = {
            "edit": op,
            "pairs": len(rows),
            "improved": sum(r["trained_margin"] > r["ordinary_margin"] for r in rows),
        }
        for side in ("ordinary", "trained"):
            m = [r[side + "_margin"] for r in rows]
            v.update(
                {
                    side + "_ranked": sum(x > 0 for x in m),
                    side + "_above_one": sum(x >= 1 for x in m),
                    side + "_median": median(m),
                    side + "_mean": mean(m),
                }
            )
        margins.append(v)
    sources = {
        w["id"]: w
        for w in read("research/sources/changed-tables/originals.json")["cohorts"][
            "train"
        ]
    }
    sql = []
    for r in unpack(R / "free-sql.json.gz"):
        x = r["result"]
        w = sources[r["id"]]
        row = {
            "id": r["id"],
            "question": w["question"],
            "ended": r["closed"],
            "output_tokens": len(r["tokens"]),
            "actual_visible_response": r["visible"],
            "statement": x["statement"],
        }
        for i, (label, v) in enumerate(
            zip(("original", "altered"), w["versions"], strict=True)
        ):
            rr = x["results"][i] if x["results"] else {}
            row.update(
                {
                    label + "_gold": v["answer"],
                    label + "_read": rr.get("used", False),
                    label + "_correct": rr.get("correct", False),
                }
            )
        row["correct_both"] = row["original_correct"] and row["altered_correct"]
        sql.append(row)
    assert (
        len(sql) == 24
        and sum(r["correct_both"] for r in sql) == decision["training_sql_both_correct"]
    )
    write_csv(P / "conditional-fit-generated-programs.csv", sql)
    jobs = [
        j
        for p in Path("research/sessions").glob("window-*.json")
        for j in read(p)["jobs"]
        if j["job"] == spec["id"]
    ]
    assert jobs
    report = {
        "decision": decision,
        "ablated_mean_gain": mean(ablations.values()),
        "ablated_improved": sum(v > 0 for v in ablations.values()),
        "fit": fit,
        "group_probes": groups,
        "balanced_gain": mean(g["gain"] for g in groups),
        "sql_margins": margins,
        "source_turns_seen": sum(r["examples"] for r in steps),
        "smallest_norm": min(r["norm"] for r in steps),
        "largest_norm": max(r["norm"] for r in steps),
        "cold_error": replay["maximum_turn_mean_abs_error"],
        "cold_positions": replay["positions"],
        "seconds": sum(j["seconds"] for j in jobs),
        "peak_gib": max(j["peak_bytes"] for j in jobs) / 2**30,
        "closed_responses": sum(r["ended"] for r in sql),
        "submitted_queries": sum(bool(r["statement"]) for r in sql),
        "both_read": sum(r["original_read"] and r["altered_read"] for r in sql),
        "original_correct": sum(r["original_correct"] for r in sql),
        "altered_correct": sum(r["altered_correct"] for r in sql),
        "query_histogram": dict(
            Counter(r["statement"] for r in sql if r["statement"]).most_common()
        ),
        "primary_eligible": decision["passed"],
    }
    save(P / "conditional-fit-study.json", report)
    plt.rcParams.update(
        {
            "svg.hashsalt": "conditional-fit-01",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.85), layout="constrained")
    bins = [steps[i : i + 32] for i in range(0, len(steps), 32)]
    a.plot(
        [g[-1]["step"] for g in bins],
        [
            sum(r["loss"] * r["targets"] for r in g) / sum(r["targets"] for r in g)
            for g in bins
        ],
        color="#236a60",
    )
    a.set(xlabel="Optimizer update", ylabel="Objective per positive token")
    for g in groups:
        b.plot(
            [g["before"], g["after"]], [g["label"], g["label"]], color="#777777", lw=1
        )
    for field, color, label in (
        ("before", "#b56437", "Ordinary instruction"),
        ("after", "#236a60", "Trained loop"),
    ):
        b.scatter(
            [g[field] for g in groups],
            [g["label"] for g in groups],
            color=color,
            label=label,
            s=15,
        )
    b.set(xlabel="Mean training probe token NLL (nats)")
    b.legend(frameon=False, fontsize=7)
    fig.savefig(P / "conditional-fit-study.svg", metadata={"Date": None})
    plt.close(fig)
    pdf = P / "conditional-fit-study.pdf"
    pdf.write_bytes(typst.compile(str(P / "conditional-fit-study.typ"), timestamp=0))
    bundle = P / "loopqwen-conditional-study.pdf"
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Author": "LoopQwen-MLX",
            "/Title": "Training an instruction-initialized recurrent decoder",
        }
    )
    for title, filename in (
        ("Native training and executed programs", pdf),
        (
            "Architecture, data and physical verification",
            P / "conditional-feature-study.pdf",
        ),
        (
            "Original Base-rooted results and methodology",
            P / "loopqwen-four-layer-study.pdf",
        ),
    ):
        src = PdfReader(filename)
        assert all(len(page.extract_text()) > 160 for page in src.pages)
        writer.append(src, outline_item=title)
    with bundle.open("wb") as f:
        writer.write(f)
    print(len(PdfReader(bundle).pages), digest(bundle), flush=True)


if __name__ == "__main__":
    build()
