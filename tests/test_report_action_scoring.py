"""Numeric endpoints must handle echoed wheel names and distinguish coincidental calculations."""

from loopqwen.calculator_tool import tool
from loopqwen.research.report_actions import score


def test_coincidental_and_operand_sensitive_calculations():
    row = {"family": "report", "expression": "19 + 23", "gold": "42"}
    for steps, equivalent in [
        (["23+19"], True),
        (["21*2"], False),
        (["19+1", "r1+22"], False),
    ]:
        event = {
            "call": {"name": "calculate", "arguments": {"steps": steps}},
            "observation": tool().callback(steps),
        }
        result = score(row, 0, {"answer": "42", "status": "answered", "turns": [event]})
        assert result["success"] and result["bound_operands"] == equivalent


def test_distribution_filename_and_missing_execution():
    file = "wheel-3.11.9-cp312.whl"
    row = {
        "questions": [f"Filename | Bytes\nList\n{file} | 23\n\nHow big?"],
        "family": "distribution",
        "expression": None,
        "gold": "23",
    }
    tr = {
        "answer": f"The size is 23 bytes for {file}.",
        "status": "answered",
        "turns": [],
    }
    assert score(row, 0, tr)["success"]
    derived = score(dict(row, expression="21+2"), 0, tr)
    assert derived["correct_answer"] and not derived["success"]
