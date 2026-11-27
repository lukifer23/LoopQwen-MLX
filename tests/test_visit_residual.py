import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.visit_residual import VisitResidualModel
from loopqwen.worker import acquire
from test_future_refinement import tiny_text_decoder


def test_zero_initial_logits_and_two_independent_native_histories():
    acquire()
    mx.random.seed(5321)
    base = LoopedModel(tiny_text_decoder(), Span(3, 1, 2, "stabilized"), alpha=0.35)
    net = VisitResidualModel(base, 8)
    base.eval()
    net.eval()
    ids = mx.array([[13, 23, 11, 71, 8]])
    assert mx.array_equal(base(ids), net(ids)).item()
    assert set(dict(tree_flatten(net.trainable_parameters()))) == {
        "refinement." + k + ".weight" for k in ("context", "change", "output")
    }
    # Turn on a nonzero readout, then compare incremental decoding to the full window.
    net.refinement.output.weight = mx.random.normal(net.refinement.output.weight.shape)
    full = net(ids)
    kv = net.make_cache()
    stepwise = mx.concatenate(
        [net(ids[:, i : i + 1], kv) for i in range(ids.shape[1])], axis=1
    )
    assert mx.allclose(full, stepwise, atol=3e-5, rtol=1e-5).item()
    assert kv.slots[3, 0].offset == kv.slots[3, 1].offset == ids.shape[1]
    assert not mx.array_equal(kv.slots[3, 0].state[0], kv.slots[3, 1].state[0]).item()
    x, y = net.visits(ids)
    assert mx.array_equal(net.refinement(x, x), x).item()
    assert mx.array_equal(net.refinement.apply_change(x, y, mx.zeros_like(y)), y).item()
    assert not mx.array_equal(net.refinement(x, y), y).item()
