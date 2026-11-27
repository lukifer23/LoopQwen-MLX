import mlx.core as mx
from loopqwen.frozen_memory_attention import frozen_attention
from loopqwen.shared_query import differentiable_attention
from loopqwen.worker import acquire


def test_frozen_memory_adjoint_all_heads_chunks_and_single_or_causal_suffix():
    acquire()
    mx.random.seed(93)
    k = mx.random.normal((2, 2, 31, 32))
    v = mx.random.normal(k.shape)
    for length, mask in ((1, None), (6, "causal")):
        q = mx.random.normal((2, 8, length, 32))
        cot = mx.random.normal(q.shape)

        def ref(z):
            return (differentiable_attention(z, k, v, 32**-0.5, mask) * cot).sum()

        def streamed(z):
            return (frozen_attention(z, k, v, 32**-0.5, mask, chunk_size=3) * cot).sum()

        a, da = mx.value_and_grad(ref)(q)
        b, db = mx.value_and_grad(streamed)(q)
        assert mx.allclose(a, b, rtol=1e-5, atol=1e-5).item()
        assert mx.allclose(da, db, rtol=3e-5, atol=2e-6).item()
