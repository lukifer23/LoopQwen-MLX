"""Construct causal action and delivery targets from two executed calculations on each report."""

from collections import Counter
from pathlib import Path

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.content_read_material import persist_rows
from scripts.crossed_numeric_sources import DIR
from scripts.program_language_material import answer
from scripts.source_copy_material import TOKENS
from scripts.table_policy_material import persist

ROOT = Path("research/sources/receipt-repair")
MATERIAL = ROOT / "targets.json.gz"
SEED = 215791


def build():
    original = unpack(DIR / "training.json.gz")
    positive = {r["question"]: r for r in original if r["condition"] == "correct"}
    swapped = {r["question"]: r for r in original if r["condition"] == "crossed"}
    pilots = set(read(DIR / "inventory.json")["pilot_reports"])
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    for key, r in sorted(positive.items()):
        wrong = swapped[key]
        other = positive[wrong["donor"]]
        assert r["observed"] == calculate(r["correct_steps"])
        assert (
            other["observed"] == wrong["observed"] == calculate(other["correct_steps"])
        )
        assert r["observed"]["value"] != other["observed"]["value"]
        question, action, receipt = r["messages"]
        wrong_action = other["messages"][1]
        correct_answer = answer(r["reference"])
        wrong_answer = answer(other["reference"])
        for kind, history, target, alternatives in (
            ("open", [question], action, [wrong_action]),
            ("deliver", r["messages"], correct_answer, [wrong_answer, action]),
            ("repair", wrong["messages"], action, [wrong_answer, wrong_action]),
            (
                "recovered",
                [*wrong["messages"], action, receipt],
                correct_answer,
                [wrong_answer, action],
            ),
        ):
            row = dict(messages=history, tools=r["tools"])
            p, y = token_pair(tok, dict(**row, target=target))
            if kind in ("deliver", "repair"):
                assert (
                    fingerprint(p)
                    == (r if kind == "deliver" else wrong)["input_sha256"]
                )
            negatives = []
            for alt in alternatives:
                q, z = token_pair(tok, dict(**row, target=alt))
                assert p == q and z != y
                pos = next(i for i, (a, b) in enumerate(zip(y, z)) if a != b)
                negatives.append([pos, z[pos]])
            rows.append(
                dict(
                    id=key + "/" + kind,
                    group=kind,
                    cluster=r["report"],
                    split="pilot" if r["report"] in pilots else "train",
                    p=p,
                    y=y,
                    negatives=negatives,
                )
            )
    previous = unpack(TOKENS)
    for group in (
        "reasoning",
        "conversation_read",
        "reuse",
        "redirect",
        "retention",
        "teacher_action",
        "teacher_delivery",
    ):
        options = sorted(
            (r for r in previous if r["group"] == group),
            key=lambda r: fingerprint([SEED, r["id"]]),
        )
        for i, r in enumerate(options[:40]):
            rows.append(
                dict(
                    **r,
                    negatives=[],
                    split="pilot" if i < min(8, len(options) // 3) else "train",
                )
            )
    assert len({r["id"] for r in rows}) == len(rows)
    rows.sort(key=lambda r: fingerprint([SEED, r["id"]]))
    ROOT.mkdir(parents=True, exist_ok=True)
    persist_rows(MATERIAL, rows)
    persist(
        ROOT / "inventory.json",
        dict(
            reports=len(positive) // 2,
            pilots=sorted(pilots),
            tokens=digest(MATERIAL),
            source=digest(DIR / "training.json.gz"),
            rehearsal=digest(TOKENS),
            counts=dict(Counter(r["split"] + "/" + r["group"] for r in rows)),
            training_positions=sum(len(r["y"]) for r in rows if r["split"] == "train"),
            distinction="The financial pilot reports are withheld only from this intervention. Their source material appeared in earlier project training. These are development diagnostics. The product comparison requires new independent sources.",
        ),
    )


if __name__ == "__main__":
    build()
