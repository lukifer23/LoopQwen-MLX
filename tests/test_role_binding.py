import mlx.core as mx

from loopqwen.research.reading_pairs import MatchedExamples
from loopqwen.research.role_binding import grade


def test_individual_margin_penalizes_a_context_even_when_its_sibling_is_confident():
    sampler = MatchedExamples.__new__(MatchedExamples)
    logits = mx.array([[10.0, 0.0], [2.0, 0.0]])
    targets = mx.array([[0, 1, 0, 1]])
    pooled = sampler.loss(logits, targets).item()
    independent = sampler.loss(logits, targets, individual=True).item()
    assert pooled < 0.01 and independent > 1.5
    fixed = logits.at[1, 1].add(5.0)
    assert sampler.loss(fixed, targets, individual=True).item() < independent


def test_table_grounding_requires_the_requested_role_and_the_actual_return():
    args = {
        "document": "people.csv",
        "key_column": "City",
        "key": "London",
        "value_column": "Visitor",
    }
    row = {"family": "table", "gold": "Ada", "arguments": args}
    turn = {
        "call": {"name": "csv_lookup", "arguments": args},
        "observation": {"value": "Ada"},
    }
    tr = {"status": "answered", "answer": "Ada was the visitor.", "turns": [turn]}
    assert grade(row, 0, tr) == {
        "completed": True,
        "correct_answer": True,
        "exact": False,
        "observed": True,
        "success": True,
    }
    tr["answer"] = "Canada"
    assert not grade(row, 0, tr)["correct_answer"]
    tr["answer"] = "Ada"
    turn["observation"] = {"value": "Eve"}
    assert grade(row, 0, tr)["exact"] and not grade(row, 0, tr)["success"]
