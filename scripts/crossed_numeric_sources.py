"""Cross successful real calculations between two original questions on the same financial report."""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.content_read_material import persist_rows
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

SOURCE = Path("research/sources/content-read/training.json.gz")
DIR = Path("research/sources/crossed-report-calculations")
SEED = 997021


def replay(w):
    _, action, returned = w["messages"]
    (fc,) = action["tool_calls"]
    args = fc["function"]["arguments"]
    assert fc["function"]["name"] == "calculate"
    receipt = calculate(args["steps"])
    assert json.loads(returned["content"]) == receipt
    return receipt


def build():
    committed(SOURCE)
    data = unpack(SOURCE)
    clusters = defaultdict(list)
    for w in data:
        if w["group"] == "quantity":
            clusters[w["cluster"]].append(w)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    for doc, qs in sorted(clusters.items()):
        a, b = sorted(qs, key=lambda w: w["id"])
        for original, other in ((a, b), (b, a)):
            true = replay(original)
            foil = replay(other)
            assert Fraction(true["value"]) != Fraction(foil["value"])
            assert original["messages"][0] != other["messages"][0]
            for good in (True, False):
                supplied = original if good else other
                messages = [original["messages"][0], *supplied["messages"][1:]]
                ids = tok.encode(
                    render(tok, messages, original["tools"]), add_special_tokens=False
                )
                rows.append(
                    dict(
                        report=doc,
                        question=original["id"],
                        donor=supplied["id"],
                        condition="correct" if good else "crossed",
                        messages=messages,
                        tools=original["tools"],
                        reference=original["target"]["content"],
                        true=true,
                        observed=true if good else foil,
                        correct_steps=original["messages"][1]["tool_calls"][0][
                            "function"
                        ]["arguments"]["steps"],
                        input_sha256=fingerprint(ids),
                        input_tokens=len(ids),
                    )
                )
    assert len(clusters) == 160 and len(rows) == 640
    selected = sorted(clusters, key=lambda c: fingerprint([SEED, c]))[:16]
    rows.sort(key=lambda r: fingerprint([SEED, r["question"], r["condition"]]))
    assert len({r["input_sha256"] for r in rows}) == len(rows)
    DIR.mkdir(exist_ok=True, parents=True)
    persist_rows(DIR / "training.json.gz", rows)
    persist(
        DIR / "inventory.json",
        dict(
            source=digest(SOURCE),
            data=digest(DIR / "training.json.gz"),
            reports=len(clusters),
            questions=2 * len(clusters),
            contexts=len(rows),
            seed=SEED,
            pilot_reports=selected,
            additional_reports=len(clusters) - len(selected),
            method="Training-only controlled intervention. Two original TAT-QA questions on each complete training report already have distinct genuine annotated programs and target quantities. Execute both with the actual public calculator. Present each question once after its own correct observation and once after the other question's valid but wrong calculation. The original user's question never changes between the two histories. No test report, model prediction or fabricated error is used. The swap is experimental rather than human-authored; the report, questions and underlying successful calculations are original and independently replayed.",
        ),
    )


if __name__ == "__main__":
    build()
