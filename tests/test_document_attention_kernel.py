import mlx.core as mx

from loopqwen.kernels.frozen_document import document_attention
from loopqwen.shared_query import differentiable_attention
from loopqwen.worker import acquire


def test_document_moments_match_grouped_causal_attention_and_query_gradient():
    acquire()
    mx.random.seed(129)
    for dim, dtype in ((32, mx.float32), (256, mx.float16)):
        k = mx.random.normal((2, 2, 71, dim)).astype(dtype)
        v = mx.random.normal(k.shape).astype(dtype)
        for length, mask in ((1, None), (6, "causal")):
            q = mx.random.normal((2, 8, length, dim)).astype(dtype)
            cot = mx.random.normal(q.shape).astype(dtype)

            def loss(query, fn):
                a = fn(query, k, v, dim**-0.5, mask)
                b = fn(query + a, k, v, dim**-0.5, mask)
                return (b.astype(mx.float32) * cot).sum()

            a, da = mx.value_and_grad(lambda z: loss(z, differentiable_attention))(q)
            b, db = mx.value_and_grad(
                lambda z: loss(z, lambda *args: document_attention(*args, tile=7))
            )(q)
            assert mx.allclose(
                a, b, atol=0.03 if dtype == mx.float16 else 0.0002, rtol=0.001
            ).item()
            assert mx.allclose(
                da, db, atol=0.0015 if dtype == mx.float16 else 0.000005, rtol=0.001
            ).item()
