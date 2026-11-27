import pytest

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.program_questions import score


def test_exact_decimal_and_references():
    assert calculate(["0.1 + 0.2", "r1 * 10", "r2 / 2"]) == {
        "results": ["3/10", "3", "3/2"],
        "value": "3/2",
    }


@pytest.mark.parametrize("steps", [["r1+1"], ["3**3"], ["sqrt(9)"], ["7/0"]])
def test_unknown_or_unsupported_operation(steps):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(steps)


def test_word_problem_scores_natural_answer_and_executed_program_independently():
    row = {"gold": "12"}
    tr = {"answer": "The total is 12 pencils.", "status": "answered", "turns": []}
    assert score(row, tr)["correct_answer"] and not score(row, tr)["success"]
    tr["turns"] = [
        {"observation": calculate(["4*3"]), "call": {"arguments": {"steps": ["4*3"]}}}
    ]
    assert score(row, tr)["success"]
    tr["turns"] = [
        {"observation": calculate(["12"]), "call": {"arguments": {"steps": ["12"]}}}
    ]
    assert not score(row, tr)["success"]


def test_checked_calculator_validates_each_side_using_the_original_decimal():
    from loopqwen.calculator_tool import checked_calculate

    assert checked_calculate(["24 / 3 = 8", "r1 * 4 = 16+16 = 32"])["value"] == "32"
    assert (
        checked_calculate(["0.100000000000000001 - 0.1 = 1/1000000000000000000"])[
            "value"
        ]
        == "1/1000000000000000000"
    )
    for step in [
        "2+3=6",
        "0.100000000000000001=0.1",
        "5==5",
        "5=",
        "=5",
        "5=r1",
        "5=x",
        "5=1/0",
    ]:
        with pytest.raises((ValueError, ZeroDivisionError)):
            checked_calculate([step])
    with pytest.raises(ValueError):
        calculate(["1+2=3"])
