from loopqwen.evaluation.extractive import answer, score
from loopqwen.research.shared_budget import vote


def test_visible_extractive_first_turn():
    tokens = [14, 248069, 42, 248046]
    text = "reason\n</think>\nFinal answer: The Rhine.<|im_end|>"
    assert score(answer(tokens, text), ["Rhine"]) == (True, 1)
    assert score("lower Rhine", ["The Rhine"]) == (False, 2 / 3)
    assert answer(tokens[:-1], text) is None
    assert answer([248068, *tokens], text) is None
    assert score(None, ["the"]) == (False, 0)


def test_shared_budget_earliest_tie_and_mathematical_lock():
    assert vote([None, None], 0) == (None, False)
    assert vote(["3", "4", "4", "3"], 0) == ("3", False)
    assert vote(["3", "3", None], 2) == ("3", False)
    assert vote(["3", None, "3"], 1) == ("3", True)
