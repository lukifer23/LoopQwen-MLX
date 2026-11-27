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


def test_rehearsal_only_projection_matches_full_vocabulary_loss_and_gradients():
    from mlx import nn
    from loopqwen.research.visit_residual_fit import gradient_function

    acquire()
    mx.random.seed(114)
    base = LoopedModel(tiny_text_decoder(), Span(3, 1, 2, "stabilized"), alpha=0.35)
    net = VisitResidualModel(base, 8)
    net.eval()
    net.refinement.output.weight = (
        mx.random.normal(net.refinement.output.weight.shape) * 0.1
    )
    ids = mx.array([[4, 13, 61, 27, 45, 11, 8, 16]])
    x, y = (z[0] for z in net.visits(ids))
    targets = mx.array([13, 61, 27, 45, 11, 8, 16, 32])
    visible = mx.array([False, True, False, True, True, True, False, True])
    anchor = mx.arange(8) >= 5
    positions, foils = mx.array([1, 3]), mx.array([23, 17])

    def full(head):
        z = net.project(net.text.model.norm(head(x, y))).astype(mx.float32)
        donor = net.project(net.text.model.norm(y)).astype(mx.float32)
        log_d = nn.log_softmax(donor, axis=-1)
        divergence = (mx.exp(log_d) * (log_d - nn.log_softmax(z, axis=-1))).sum(-1)
        ce = nn.losses.cross_entropy(z, targets)
        active_visible = visible * (~anchor)
        primary = (
            0.35 * ce[:5].mean()
            + 0.65 * (ce * active_visible).sum() / active_visible.sum()
        )
        margins = z[positions, targets[positions]] - z[positions, foils]
        ranking = mx.logaddexp(mx.zeros_like(margins), 1.0 - margins).mean()
        return (
            0.45 * primary
            + 0.15 * ranking
            + 0.2 * ce[5:].mean()
            + 0.2 * (divergence * anchor).sum() / anchor.sum()
        )

    expected, expected_grad = nn.value_and_grad(net.refinement, full)(net.refinement)
    parts, received_grad = gradient_function(net)(
        net.refinement, x, y, targets, visible, anchor, y[5:], positions, foils
    )
    assert mx.allclose(parts[0], expected, atol=2e-6).item()
    a, b = dict(tree_flatten(expected_grad)), dict(tree_flatten(received_grad))
    assert a.keys() == b.keys()
    for key in a:
        assert mx.allclose(a[key], b[key], atol=2e-6, rtol=2e-5).item()


def test_dependency_verdict_preserves_wildcards_and_distinguishes_markdown_emphasis():
    from loopqwen.research.visit_residual_study import score

    event = {
        "call": {
            "name": "wheel_metadata",
            "arguments": {"wheel": "demo.whl", "field": "Requires-Dist"},
        },
        "observation": {"values": ["library>=3.10"]},
    }
    row = {
        "family": "dependency",
        "current": event,
        "requirement": ">=3.10",
        "allowed": True,
    }

    def grade(text):
        return score(
            None, row, 0, {"status": "answered", "answer": text, "turns": [event, {}]}
        )

    v = grade("The requirement is **>=3.10**. Yes.")
    assert v["success"] and not v["literal_range"]
    assert not grade(">=3.10 or >=3.9. Yes.")["success"]
    assert not grade(">=3.10. No.")["success"]
    row["requirement"] = "==1.*"
    assert grade("The allowed releases are **==1.***. Yes.")["success"]
