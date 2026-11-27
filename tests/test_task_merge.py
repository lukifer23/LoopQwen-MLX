import mlx.core as mx

from loopqwen.research.task_merge import consensus
from loopqwen.worker import acquire


def test_elects_by_magnitude_averages_agreeing_updates_and_keeps_exact_ties_at_zero():
    acquire()
    a = mx.array([3.0, 1.0, 0.05, 3.0, 4.0])
    b = mx.array([-2.0, 1.0, 0.08, -3.0, 5.0])
    c = mx.array([2.0, -3.0, 0.01, 0.0, -1.0])
    combined, report = consensus([a, b, c], 0.8)
    assert mx.allclose(combined, mx.array([2.5, -3.0, 0.01, 0.0, 4.5])).item()
    assert sum(report["participation"]) == 5 and report["nonzero"] == 4


def test_zero_deltas_leave_all_weights_unchanged():
    acquire()
    x = mx.zeros((4, 2))
    result, record = consensus([x, x], 0.2)
    assert mx.array_equal(x, result).item() and record["nonzero"] == 0


def test_native_shared_update_uses_both_actual_visits_and_has_a_finite_gradient():
    from mlx import nn
    from mlx.utils import tree_flatten
    from test_future_refinement import tiny_text_decoder

    from loopqwen.research.integrated_assistance_fit import learning_objective
    from loopqwen.tool_model import tool_view

    acquire()
    mx.random.seed(4171)
    net = tool_view(tiny_text_decoder(), 3, 4, 2)
    ids = mx.array([[4, 17, 21, 72, 38]])
    hidden = net.prefix_hidden(ids)
    indices = mx.array([0, 1, 3])
    labels = mx.array([17, 21, 38])
    (total, second, first, ranking), grad = nn.value_and_grad(net, learning_objective)(
        net, hidden, indices, labels
    )
    loss_from_serving = nn.losses.cross_entropy(net(ids)[0, indices], labels).mean()
    assert mx.allclose(loss_from_serving, second, atol=1e-5).item()
    assert (
        mx.allclose(total, second + 0.15 * ranking).item()
        and first.item() != second.item()
    )
    gradients = [v for _, v in tree_flatten(grad)]
    assert len(gradients) == 14 and all(mx.isfinite(v).all().item() for v in gradients)
    assert any(mx.any(v != 0).item() for v in gradients)
