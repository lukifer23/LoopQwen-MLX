import random

import pytest

from loopqwen.data.tokenize import physical_batches
from loopqwen.data.transitions import render, states
from loopqwen.data.v4 import dag, pointer
from loopqwen.evaluation.trace_verifier import verify


def test_physical_position_cap_preserves_every_document():
    rows = [
        {"ids": list(range(n + 1)), "id": i} for i, n in enumerate((91, 100, 101, 120))
    ]
    groups = list(physical_batches(rows, 360))
    assert [r for g in groups for r in g] == rows
    assert list(map(len, groups)) == [3, 1]
    assert all(
        max(16, max(len(r["ids"]) - 1 for r in g)) * len(g) <= 360 for g in groups
    )
    with pytest.raises(ValueError):
        list(physical_batches(rows, 98))


def test_pointer_labels_are_absent_from_prompt_construction():
    row = dict(id="map", map=[1, 2, 3, 4, 5, 6, 7, 0], start=3)
    prompt = render(row, 3)
    row["states"] = states(row["map"], row["start"], 6)
    assert render(row, 3) == prompt
    assert row["states"] == [4, 5, 6, 7, 0, 1]
    task = pointer(row, 3)
    assert verify(task, task["trace"])["verified"]
    altered = verify(task, task["trace"].replace("Next(4) = 5", "Next(4) = 6"))
    assert (
        altered["final"] == "correct"
        and "incorrect_state_transition" in altered["checks"]
    )


def test_v4_graph_certificate_needs_prerequisites():
    for shift in ("id", "operand", "topology", "combined"):
        graph = dag(random.Random(19), shift)
        assert verify(graph, graph["trace"])["verified"]
        assert not verify(graph, graph["trace"].splitlines()[-1])["verified"]
