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


def test_weighted_joint_checkpoint_restores_both_optimizers(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from loopqwen import checkpoint
    from loopqwen.config import config
    from loopqwen.data.v2 import rows
    from loopqwen.data.tokenize import batch
    from loopqwen.training.losses import exit_objective
    from loopqwen.training.optimizers import optimizer

    c = config("configs/stabilized_loop.yaml")
    c.update(gate_lr=0.001, first_exit_weight=0.25)
    model, _, _ = checkpoint.setup(c)
    model.train()
    opt = optimizer(c)
    ids, positions, targets = batch(rows("validation", "answer")[:2])
    weights = mx.where(mx.arange(targets.size) % 2 == 0, 0.25, 2.0)

    def objective(m):
        total, components, raw = exit_objective(
            m, ids, positions, targets, weights, 0.25
        )
        return total / weights.sum(), components / weights.sum(), raw / targets.size

    (loss, parts, raw), grad = nn.value_and_grad(model, objective)(model)
    mx.eval(loss, parts, raw, grad)
    assert abs(loss.item() - (0.25 * parts[0] + 0.75 * parts[1]).item()) < 1e-5
    assert raw.shape == (2,) and 0 < abs(grad["alpha_logit"].item()) < 1e3
    assert any(
        mx.any(v != 0).item() for k, v in tree_flatten(grad) if k.endswith("lora_b")
    )
    opt.update(model, grad)
    mx.eval(model.trainable_parameters(), opt.state)
    checkpoint.write(tmp_path, model, opt, dict(step=1, config=c))

    def snapshot(m, o):
        return dict(tree_flatten(m.trainable_parameters())), dict(tree_flatten(o.state))

    opt.update(model, grad)
    expected = snapshot(model, opt)
    fresh = optimizer(c)
    checkpoint.restore(tmp_path, model, fresh)
    fresh.update(model, grad)
    for got, want in zip(snapshot(model, fresh), expected, strict=True):
        assert got.keys() == want.keys()
        assert all(mx.array_equal(v, want[k]).item() for k, v in got.items())
