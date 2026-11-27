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
