import mlx.core as mx

from loopqwen.research.distill_training import bucket_kl, positions
from loopqwen.research.task_vectors import factors


def test_projected_linear_update_and_padding_reconstruct_a_known_matrix():
    mx.random.seed(17)
    for out, inp in ((7, 16), (37, 26)):
        delta = mx.random.normal((out, 3)) @ mx.random.normal((3, inp))
        a, b, energy = factors(delta, 8, 2, 0.85)
        assert a.shape == (inp, 8) and b.shape == (8, out)
        assert mx.max(mx.abs(0.85 * delta - b.T @ a.T)).item() < 0.0001
        assert abs(energy.item() - 1) < 0.0001


def test_teacher_buckets_include_residual_mass_and_score_response_positions_only():
    z = mx.array([[0.3, -0.5, 1.0]])
    ids = mx.array([[0, 2]])
    mass = mx.take_along_axis(mx.softmax(z, axis=-1), ids, axis=-1)
    assert abs(bucket_kl(z, ids, mass, 1.0).item()) < 1e-6
    assert bucket_kl(-z, ids, mass, 1.0).item() > 0.1
    d = {"ids": list(range(10)), "prefix": 6}
    assert positions(d, False, 12) == [5, 6, 7, 8]
    assert positions(d, True, 3) == [0, 4, 8]
