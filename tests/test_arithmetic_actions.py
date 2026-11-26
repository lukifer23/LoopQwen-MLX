from loopqwen.arithmetic_actions import as_steps, mentions
from loopqwen.calculator_tool import execute


def test_signed_rational_operands_and_intermediate_references():
    text = "The readings were -3.5 and 1.5/2. The inventory--38, with a dozen samples."
    values = [s[2] for s in mentions(text)]
    assert values == ["-7/2", "3/4", "38", "12"]
    steps = as_steps(values, [["-", "n1", "n0"], ["*", "r0", "c1"]])
    assert steps == ["(3/4)-(-7/2)", "r1*2"]
    assert str(execute(steps)[-1]) == "17/2"


def test_zero_denominator_in_a_record_is_two_observed_quantities():
    text = "The old record was 0/0."
    assert [(text[a:b], v) for a, b, v in mentions(text)] == [("0", "0"), ("0", "0")]
