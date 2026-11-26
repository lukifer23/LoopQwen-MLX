"""Audit local training/evaluation lexical overlap without filtering any reported scores."""

import csv
import re
from collections import defaultdict
from pathlib import Path

from loopqwen.io import read

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data"
TYPED = "tool-dialogue-typed-training"


def rows(name):
    obj = read(DATA / (name + ".json"))
    return obj.get("examples", obj.get("rows"))


def text(row, field):
    if "turns" in row:
        value = "\n".join(
            ("Visitor" if t["speaker"] == "USER" else "Agent") + ": " + t["utterance"]
            for t in row["turns"]
        )
    elif "question" in row:
        value = row["question"]
        if isinstance(value, list):
            value = value[0][0]["content"]
    else:
        value = next(x["content"] for x in row["messages"] if x["role"] == "user")
    if field in ("passage", "boolq_question"):
        pair = value.removeprefix("Passage\n").split("\n\nQuestion\n", 1)
        if len(pair) != 2:
            return ""
        value = pair[0 if field == "passage" else 1]
    if field == "dialogue":
        if not value.startswith("Visitor: "):
            return ""
        value = value.split("\n\nUse the table to count the venues", 1)[0]
    if field == "program":
        value = value.split("\nUse the calculator for the arithmetic", 1)[0]
    if field in ("sqlite_question", "public_retention"):
        from loopqwen.evaluation.table_questions import INSTRUCTION

        value = value.removesuffix(INSTRUCTION)
    return value


def normalize(s, mask):
    words = re.findall(r"[^\W_]+", s.casefold())
    return tuple("<number>" if mask and w.isdecimal() else w for w in words)


def shingle(words):
    return {words[i : i + 5] for i in range(len(words) - 4)}


def audit(train, held, mask):
    bank = {normalize(x, mask) for x in train}
    queries = {normalize(x, mask) for x in held}
    grams = [shingle(x) for x in sorted(bank)]
    posting = defaultdict(list)
    for i, gs in enumerate(grams):
        for g in gs:
            posting[g].append(i)
    exact = sum(q in bank for q in queries)
    similar = 0
    worst = 0.0
    for q in queries:
        qg = shingle(q)
        indices = set(i for g in qg for i in posting.get(g, []))
        best = max(
            (len(grams[i] & qg) / len(grams[i] | qg) for i in indices), default=0.0
        )
        similar += best >= 0.6 and q not in bank
        worst = max(worst, best)
    return [len(bank), len(queries), exact, similar, worst]


def build():
    studies = [
        (
            "fresh competing schemas after rehearsal",
            [TYPED, "tool-counterfactual-training"],
            "tool-counterfactual-bfcl",
            "new_schema",
        ),
        (
            "operand planner external questions",
            ["operand-planner-training"],
            "operand-planner-test",
            "question",
        ),
        ("tool public executions", [TYPED], "tool-exec-bfcl-01", "question"),
        ("tool multiple functions", [TYPED], "tool-bfcl-multiple-01", "question"),
        ("tool single-function recovery", [TYPED], "tool-bfcl-math-01", "question"),
        ("WikiSQL cold transfer", [TYPED], "tool-table-queries-01", "question"),
        (
            "WikiSQL supervised transfer",
            [TYPED, "tool-sql-learning-training"],
            "tool-sql-learning-panel",
            "sqlite_question",
        ),
        (
            "WikiSQL fit public retention",
            [TYPED, "tool-sql-learning-training"],
            "tool-sql-learning-panel",
            "public_retention",
        ),
        (
            "calculator word problems",
            [TYPED, "tool-program-reasoning-training"],
            "tool-program-reasoning-01",
            "program",
        ),
        (
            "HotpotQA bridge",
            ["bridge-reading-train"],
            "bridge-reading-validation",
            "question",
        ),
        (
            "MultiWOZ state",
            ["tool-venue-dialogue-training"],
            "tool-venue-dialogue-test",
            "dialogue",
        ),
    ]
    for tag, t, h in [
        ("first", "contrast-boolq-training", "contrast-boolq-panel"),
        ("second", "contrast-boolq-transfer-training", "contrast-boolq-transfer-panel"),
    ]:
        for f in ("passage", "boolq_question"):
            studies.append(("BoolQ " + tag, [t], h, f))
    with (ROOT / "paper/training-overlap.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "study",
                "field",
                "number_mask",
                "unique_training",
                "unique_heldout",
                "exact",
                "additional_jaccard_at_least_0.60",
                "maximum_five_word_jaccard",
            ]
        )
        for study, ts, held, field in studies:
            train = {text(r, field) for tr in ts for r in rows(tr)} - {""}
            queries = {
                text(r, field)
                for r in rows(held)
                if (field != "dialogue" or "turns" in r)
                and (field != "new_schema" or r["panel"] == "fresh")
                and (field != "sqlite_question" or r["provider"] == "wikisql")
                and (field != "public_retention" or r["provider"] == "bfcl_exec")
            } - {""}
            for mask in (False, True):
                w.writerow([study, field, mask, *audit(train, queries, mask)])


def reading_sensitivity():
    from loopqwen.evaluation.mechanism import unpack
    from contrast_study import summarize

    with (ROOT / "paper/reading-overlap-sensitivity.csv").open("w") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "study",
                "flagged_passage_id",
                "source_training_id",
                "remaining_pairs",
                "candidate_both",
                "base_both",
                "candidate_probe",
                "base_probe",
                "pair_wins",
                "pair_losses",
                "p_exact",
            ]
        )
        for name, train, held in [
            (
                "contrast-boolq-recurrent-01",
                "contrast-boolq-training",
                "contrast-boolq-panel",
            ),
            (
                "contrast-boolq-transfer-01",
                "contrast-boolq-transfer-training",
                "contrast-boolq-transfer-panel",
            ),
        ]:
            training = [
                (r["id"], shingle(normalize(text(r, "passage"), False)))
                for r in rows(train)
            ]
            evaluated = rows(held)
            exclusions = {}
            for row in evaluated:
                query = shingle(normalize(text(row, "passage"), False))
                for tid, gs in training:
                    if gs and len(query & gs) / len(query | gs) >= 0.6:
                        exclusions[row["pair"]] = tid
            folder = ROOT / "research/results" / name
            spec = read(folder / "registration.json")
            index = [i for i, r in enumerate(evaluated) if r["pair"] not in exclusions]
            filtered = {}
            for side in ("candidate", "control"):
                outputs = unpack(folder / (side + ".json.gz"))
                filtered[side] = [outputs[i] for i in index]
            d = summarize(filtered, [evaluated[i] for i in index], spec)
            a, b = d["metrics"].values()
            w.writerow(
                [
                    name,
                    "|".join(sorted(exclusions)),
                    "|".join(exclusions[k] for k in sorted(exclusions)),
                    d["pairs"],
                    a["pairs"],
                    b["pairs"],
                    a["probe"],
                    b["probe"],
                    d["pair_gains"],
                    d["pair_losses"],
                    d["p_exact"],
                ]
            )


if __name__ == "__main__":
    build()
    reading_sensitivity()
