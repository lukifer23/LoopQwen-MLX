"""Measure arithmetic planning and completed numeric content separately."""

import re
from fractions import Fraction

from ..calculator_tool import tool
from ..evaluation.answers import NUMBER, numeric

INSTRUCTION = (
    "\nUse the calculator for the arithmetic and give the final numeric answer."
)


def tools_for(_):
    return [tool()]


def score(row, trajectory):
    gold = Fraction(row["gold"])
    calls = [t for t in trajectory["turns"] if "observation" in t]
    grounded = False
    if calls and "value" in calls[-1]["observation"]:
        grounded = Fraction(calls[-1]["observation"]["value"]) == gold and any(
            re.search(r"[+*/]|(?<=[0-9r)])\s*-", s)
            for s in calls[-1]["call"]["arguments"]["steps"]
        )
    visible = trajectory["answer"] or ""
    strict = numeric(visible)
    answer = strict
    if answer is None:
        # GSM-style last number credits a clear final quantity embedded in prose.
        matches = re.findall(rf"(?:{NUMBER})(?:\s*/\s*(?:{NUMBER}))?", visible)
        if matches:
            answer = numeric(matches[-1])
    correct = answer is not None and Fraction(answer) == gold
    complete = trajectory["status"] == "answered"
    return {
        "success": grounded and correct and complete,
        "path": grounded,
        "correct_answer": correct,
        "strict_answer": strict is not None and Fraction(strict) == gold,
        "completed": complete,
    }
