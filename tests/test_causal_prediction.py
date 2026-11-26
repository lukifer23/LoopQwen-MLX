import mlx.core as mx

from loopqwen.neural_cache import weights
from loopqwen.research.cache_language import mask


def test_continuous_memory_only_votes_with_previously_observed_next_tokens():
    similarity = mx.array([[0.2, 0.3, 0.4], [0.4, 0.9, 1.0], [0.2, 0.7, 0.9]])
    parameters = mx.array([2.0, -2.0, 1.0])
    history = mask(0, 3, 3)
    distribution, gate = weights(parameters, similarity, history)
    assert mx.all(distribution * ~history == 0).item()
    assert gate[0].item() == 0
    assert mx.allclose(distribution.sum(axis=-1), mx.array([0.0, 1.0, 1.0])).item()
    changed = mx.where(history, similarity, similarity + 500.0)
    other, other_gate = weights(parameters, changed, history)
    assert mx.array_equal(distribution, other).item()
    assert mx.array_equal(gate, other_gate).item()
    assert mask(257, 258, 258).tolist()[0][:3] == [False, True, True]
