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


def test_real_cache_rejects_changed_execution_contract():
    from dataclasses import replace
    import mlx.core as mx
    from loopqwen.checkpoint import load_trained

    model, _, _ = load_trained("adapters/phase3/p3_learn1_250k")
    tokens = mx.array([[42, 15, 89]])
    original = model.span
    cache = model.make_cache()
    mx.eval(model.hidden(tokens, cache))
    model.span = replace(original, mode="input_anchor")
    with pytest.raises(ValueError):
        model.hidden(tokens, cache)
    model.span = original
    model.alpha_override = 0.25
    with pytest.raises(ValueError):
        model.hidden(tokens, cache)
    model.alpha_override = None
    with pytest.raises(ValueError):
        model.hidden(tokens, cache, window_size=8)
    model.update({"alpha_logit": model.alpha_logit + 0.01})
    with pytest.raises(ValueError):
        model.hidden(tokens, cache)


def test_evaluation_fingerprint_covers_token_stream():
    from loopqwen.identity import fingerprint

    assert fingerprint([{"id": "same", "ids": [1, 2]}]) != fingerprint(
        [{"id": "same", "ids": [1, 3]}]
    )
