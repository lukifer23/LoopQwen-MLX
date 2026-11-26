import pytest

from loopqwen.research.unified_session_fit import supervision


def test_short_decision_and_long_worked_reply_receive_equal_loss_mass():
    yes, no, eos = 11, 12, 99
    short = [1, 2, yes, eos]
    long = [4] * 30 + [6, 21, 22, 23, eos]
    weights, positions, opposite, mask = supervision(
        [(short, 2, "yes"), (long, 31, "some words")], 0.65, yes, no
    )
    assert sum(weights[: len(short)]) == pytest.approx(0.5)
    assert sum(weights[len(short) :]) == pytest.approx(0.5)
    assert sum(weights[2:4]) == pytest.approx(0.5 * (0.65 + 0.35 * 2 / 4))
    assert weights[-1] > weights[4]
    assert positions == [2, len(short) + 31]
    assert opposite[0] == no and mask == [True, False]
