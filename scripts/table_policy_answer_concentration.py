"""Measure concentration of visible answers after the model actually reads changed data."""

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions
from loopqwen.research.table_worlds import SOURCE
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_reading import RATINGS, ROOT, committed

P = Path("paper")
LEARN = Path("research/results/table-execution-learning-01")


def run():
    committed(RATINGS)
    src = read(SOURCE)["cohorts"]
    panel = read("research/data/table-policy-panel.json")["rows"]
    logged = unpack(LEARN / "on-policy-rollouts.json.gz")
    train = {w["id"]: w for w in src["train"]}
    dev = {w["id"]: w for w in src["dev"]}
    distributions = {}
    clean = {}

    def add(label, values):
        distributions[label] = Counter(values)

    add(
        "Human training labels",
        [v["answer"] for w in train.values() for v in w["versions"]],
    )
    add("Human held labels", [v["answer"] for w in dev.values() for v in w["versions"]])
    draws = []
    sample_gold = []
    for step in logged:
        if step["stage"] != "answer":
            continue
        w = train[step["question"]]
        for v, group in zip(w["versions"], step["details"], strict=True):
            draws += [r["visible"] for r in group]
            sample_gold += [v["answer"]] * len(group)
    add("Human labels, sampled contexts", sample_gold)
    add("Policy draws after a known read", draws)
    requests = []
    pairs = defaultdict(dict)
    statements = Counter()
    for arm in ("candidate", "control"):
        visible = []
        stmts = []
        for w, record in zip(panel, unpack(ROOT / (arm + ".json.gz")), strict=True):
            if w["family"] != "changed_table":
                continue
            tr = record["requests"][0]["trajectory"]
            reads = [
                e
                for e in executions(tr)
                if e["call"]["name"] == "database_query"
                and "rows" in e["observation"]
                and e["observation"].get("referenced_tables") == ["records"]
                and not e["observation"]["truncated"]
            ]
            vals = [
                fingerprint([e["observation"]["columns"], e["observation"]["rows"]])
                for e in reads
            ]
            sql = [e["call"]["arguments"]["statement"] for e in reads]
            got = (tr["answer"] or "").strip()
            visible.append(got)
            stmts.extend(sql)
            source = next(
                v for v in dev[w["row_id"]]["versions"] if v["world"] == w["world"]
            )
            assert source["answer"] == w["gold"]
            real = fingerprint([source["rows"][0], source["rows"][1:]])
            issued = real in vals
            row = {
                "id": w["id"],
                "question": w["row_id"],
                "arm": arm,
                "world": w["world"],
                "gold": w["gold"],
                "delivered": got,
                "natural_stop": tr["status"] == "answered",
                "first_prompt": fingerprint(tr["turns"][0]["prompt_ids"]),
                "full_source": real,
                "read_full_actual_source": issued,
                "all_real_reads": len(vals),
                "query_statements": sql,
                "observed_table_digests": vals,
                "grounded_correct": record["requests"][0]["score"]["success"],
            }
            requests.append(row)
            pairs[(arm, w["row_id"])][w["world"]] = row
        add("Held loop answers" if arm == "candidate" else "Held Base answers", visible)
        statements[arm] = Counter(stmts)
    pair_rows = []
    for (arm, k), worlds in sorted(pairs.items()):
        a, b = sorted(worlds.values(), key=lambda r: r["world"] != 0)
        assert (
            a["first_prompt"] == b["first_prompt"]
            and a["full_source"] != b["full_source"]
            and a["gold"] != b["gold"]
        )
        pair_rows.append(
            {
                "arm": arm,
                "question": k,
                "both_complete_actual_reads": a["read_full_actual_source"]
                and b["read_full_actual_source"],
                "identical_answer": a["delivered"] == b["delivered"],
                "identical_nonempty": bool(a["delivered"])
                and a["delivered"] == b["delivered"],
                "both_answer_four": a["delivered"] == b["delivered"] == "4",
                "both_grounded": a["grounded_correct"] and b["grounded_correct"],
            }
        )
    write_csv(P / "table-answer-concentration-requests.csv", requests)
    write_csv(P / "table-answer-concentration-pairs.csv", pair_rows)
    freq = []
    for label, v in distributions.items():
        for name, n in v.most_common():
            freq.append(
                {
                    "population": label,
                    "literal": name,
                    "count": n,
                    "denominator": v.total(),
                }
            )
    write_csv(P / "table-answer-concentration-literals.csv", freq)
    for arm in ("candidate", "control"):
        rr = [r for r in pair_rows if r["arm"] == arm]
        observed = [r for r in rr if r["both_complete_actual_reads"]]
        clean[arm] = {
            "both_read": len(observed),
            "unchanged_after_two_reads": sum(r["identical_nonempty"] for r in observed),
            "both_four_after_two_reads": sum(r["both_answer_four"] for r in observed),
            "identical_nonempty_all": sum(r["identical_nonempty"] for r in rr),
            "queries": dict(statements[arm]),
        }
    labels = list(distributions)
    summary = {
        "populations": [
            [
                name,
                distributions[name].total(),
                distributions[name]["4"],
                len(distributions[name]),
            ]
            for name in labels
        ],
        "paired": clean,
        "native_primary": digest(ROOT / "checksums.json"),
        "training": digest(LEARN / "checksums.json"),
    }
    save(P / "table-answer-concentration.json", summary)
    plt.rcParams.update(
        {
            "svg.hashsalt": "actual-four-response",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.7, 2.45), layout="constrained")
    series = [labels[0], labels[2], labels[3], labels[1], labels[4], labels[5]]
    rates = [100 * distributions[k]["4"] / distributions[k].total() for k in series]
    ax.barh(
        range(len(series)),
        rates,
        color=["#929899", "#929899", "#478476", "#929899", "#136552", "#795488"],
    )
    ax.set(
        yticks=range(len(series)),
        yticklabels=series,
        xlabel="Share whose entire visible answer or source label is 4 (%)",
        xlim=(0, 105),
    )
    ax.invert_yaxis()
    for i, x in enumerate(rates):
        ax.text(x + 0.8, i, f"{x:.1f}%", va="center", fontsize=8)
    fig.savefig(P / "table-answer-concentration.svg", metadata={"Date": None})
    plt.close(fig)
    pdf = typst.compile(str(P / "table-answer-concentration.typ"), timestamp=0)
    (P / "table-answer-concentration.pdf").write_bytes(pdf)
    print(summary)


if __name__ == "__main__":
    run()
