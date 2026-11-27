import mlx.core as mx
from mlx.utils import tree_flatten
from test_shared_query import model

from loopqwen.bounded_read import BoundedContentRead, FRACTION


def test_zero_write_identity_and_bound_for_large_values_and_multiple_queries():
    parent = model()
    net = BoundedContentRead(parent.text, revisits=2, rank=8, gate_rank=4)
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    net.eval()
    ids = mx.array([[1, 3, 5, 7, 9, 11]])
    assert mx.array_equal(net(ids), parent(ids)).item()
    _, state, _, values, _, mask = net.encode(ids)
    net.step.write.weight = mx.random.normal(net.step.write.weight.shape) * 1000
    state = state.astype(mx.float32)
    bound = FRACTION * mx.sqrt(mx.mean(state.square(), axis=-1, keepdims=True) + 1e-6)
    for magnitude in (0.01, 10, 1000):
        new = net.step.recall(state, values * magnitude, mask, differentiate=False)
        assert mx.all(mx.abs(new - state) <= bound + 1e-6).item()
        assert mx.any(mx.abs(new - state) > bound * 0.95).item()
