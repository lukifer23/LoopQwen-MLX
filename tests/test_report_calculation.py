from loopqwen.calculator_tool import calculate
from loopqwen.research.report_calculation import grade, number


def test_grounding_requires_real_arithmetic_and_tracks_reuse_across_callbacks():
    row = {"gold": "10", "tolerance": "0", "steps": ["3+2", "5*2"]}

    def trajectory(expressions):
        return {
            "status": "answered",
            "answer": "The answer is 10.",
            "turns": [
                {
                    "call": {"name": "calculate", "arguments": {"steps": [x]}},
                    "observation": calculate([x]),
                }
                for x in expressions
            ],
        }

    result = grade(row, trajectory(["3+2", "5*2"]))
    assert (
        result["correct"]
        and result["grounded"]
        and result["linked_grounded"]
        and result["expert_values"]
    )
    result = grade(row, trajectory(["10+0"]))
    assert result["correct"] and not result["grounded"]
    assert not grade(row, trajectory(["3+2", "4*2"]))["grounded"]


def test_report_numeric_score_preserves_fraction_and_percentage_point_convention():
    from fractions import Fraction

    assert number("The increase was 12.5%.")[0] == Fraction(25, 2)
    assert number("3/8") == (Fraction(3, 8), True)
    assert number("unknown") == (None, False)
