from dataclasses import replace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel
from test_future_refinement import tiny_text_decoder

from loopqwen.conditional_reentry import ConditionalReentry
from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.research.conditional_features import changed_token_spans
from loopqwen.research.conditional_fit import objectives
from loopqwen.worker import acquire


@pytest.mark.parametrize("passes", [1, 2])
def test_native_last_position_and_matched_readout_gradients(passes):
    acquire()
    mx.random.seed(816347)
    text = TextModel(replace(tiny_text_decoder().args, num_hidden_layers=24))
    net = ConditionalReentry(text, {"passes": passes, "rank": 8, "gate_rank": 4})
    net.eval()
    ids = mx.array([[10, 7, 18, 28, 9, 4, 8, 73]])
    dense = LoopedModel(text, Span(20, 4, 1))
    assert mx.array_equal(net(ids), dense(ids)).item()
    # Under both interventions the initial upstream representation is exactly ordinary.
    features = net.features(ids)
    logits = net.project(net.text.model.norm(net.readout(features)))
    assert mx.array_equal(logits, net(ids)).item()
    assert features.shape == (1, 8, 5, 64)
    loss = objectives(net, 48, 1)
    y = ids[0, 1:]
    coef = mx.array([[0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    lossval, grad = nn.value_and_grad(net.readout, loss)(
        net.readout, features[0, :-1], y, mx.ones((7,)), coef
    )
    base = nn.losses.cross_entropy(
        logits[0, :-1].astype(mx.float32), y, reduction="none"
    )
    assert mx.allclose(
        lossval, base.sum() + 48 * nn.softplus(1 + base[1] - base[4])
    ).item()
    assert (
        mx.abs(grad["bias"]).sum().item() > 0
        and mx.abs(grad["context"]["up"]["weight"]).sum().item() > 0
    )
    assert all(mx.abs(c["up"]["weight"]).sum().item() > 0 for c in grad["corrections"])
    assert [k for k, _ in tree_flatten(net.trainable_parameters())] == [
        "readout." + k for k, _ in tree_flatten(net.readout.parameters())
    ]
    net.readout.bias = mx.full((64,), 0.32)
    cache = net.make_cache()
    net.last_hidden(ids[:, :3], cache)
    out = net.last_hidden(ids[:, 3:], cache)
    assert len(cache.slots) == 20 + 4 * passes
    assert mx.allclose(out, net.hidden(ids)[:, -1:], atol=1e-4, rtol=1e-4).item()
    assert not mx.array_equal(net(ids), dense(ids)).item()
    assert mx.allclose(
        net.features(ids[:, :4]), features[:, :4], atol=1e-4, rtol=1e-4
    ).item()


def test_sql_preference_spans_include_continuation_for_insertions():
    assert changed_token_spans([4, 9, 0], [4, 11, 0]) == ([1], [1])
    assert changed_token_spans([4, 9, 21, 0], [4, 21, 0]) == ([1, 2], [1])
    assert changed_token_spans([4, 21, 0], [4, 9, 21, 0]) == ([1], [1, 2])
