"""Exact grouped attention and a streamed query derivative for an immutable training document."""

import mlx.core as mx


def frozen_attention(q, k, v, scale, mask, *, chunk_size=2048):
    """Differentiate queries while keeping every input key and value fixed.

    The cached backbone supplies k and v as training data. FP32 chunk reductions
    cover every causal position. Only one projected score block is needed at a
    time in the query derivative, including for one-position suffixes.
    """
    b, heads, length, d = q.shape
    if (
        k.shape != v.shape
        or k.shape[0] != b
        or k.shape[-1] != d
        or heads % k.shape[1]
        or chunk_size < 1
    ):
        raise ValueError("Incompatible frozen attention shapes or chunk size")
    groups = k.shape[1]
    n = k.shape[2]
    if isinstance(mask, str) and mask != "causal":
        raise ValueError("Unknown attention mask")
    if n < length or n == 0:
        raise ValueError("The suffix must fit in the frozen history")

    def blocks(query):
        grouped = query.astype(mx.float32).reshape(
            b, groups, heads // groups, length, d
        )
        for first in range(0, n, chunk_size):
            last = min(first + chunk_size, n)
            keys = k[:, :, None, first:last].astype(mx.float32)
            scores = (grouped * scale) @ keys.swapaxes(-1, -2)
            if isinstance(mask, str):
                valid = (
                    mx.arange(first, last)[None, :] <= mx.arange(n - length, n)[:, None]
                )
                scores = mx.where(valid, scores, -mx.inf)
            elif mask is not None:
                part = mask[..., first:last]
                if part.ndim == 4:
                    part = part[:, :, None]
                scores = (
                    mx.where(part, scores, -mx.inf)
                    if part.dtype == mx.bool_
                    else scores + part
                )
            yield first, last, keys, scores

    @mx.custom_function
    def forward(query):
        # A traced short Metal SDPA can request an unsupported log-normalizer
        # even under a custom derivative when its output feeds the next visit.
        shape = (b, groups, heads // groups, length, 1)
        largest = mx.full(shape, mx.finfo(mx.float32).min)
        partition = mx.zeros(shape)
        weighted = mx.zeros((*shape[:-1], d))
        for first, last, _, scores in blocks(query):
            new_largest = mx.maximum(largest, scores.max(axis=-1, keepdims=True))
            previous = mx.exp(largest - new_largest)
            weights = mx.exp(scores - new_largest)
            values = v[:, :, None, first:last].astype(mx.float32)
            weighted = previous * weighted + weights @ values
            partition = previous * partition + weights.sum(axis=-1, keepdims=True)
            largest = new_largest
            mx.eval(weighted, partition, largest)
        result = weighted / mx.where(partition > 0, partition, 1)
        return result.reshape(query.shape).astype(query.dtype)

    @forward.vjp
    def query_gradient(query, upstream, _result):
        # For each head O=sum(p_i*v_i). The derivative with respect to a
        # logit is p_i times the inner product of upstream with (v_i-O).
        maxima = mx.full((b, groups, heads // groups, length, 1), -mx.inf)
        for _, _, _, scores in blocks(query):
            maxima = mx.maximum(maxima, mx.max(scores, axis=-1, keepdims=True))
            mx.eval(maxima)
        maxima = mx.where(mx.isfinite(maxima), maxima, 0)
        denominator = mx.zeros_like(maxima)
        moment = mx.zeros_like(maxima)
        g = upstream.astype(mx.float32).reshape(b, groups, heads // groups, length, d)
        for first, last, _, scores in blocks(query):
            weights = mx.exp(scores - maxima)
            values = v[:, :, None, first:last].astype(mx.float32)
            denominator = denominator + weights.sum(axis=-1, keepdims=True)
            moment = moment + (weights * (g @ values.swapaxes(-1, -2))).sum(
                axis=-1, keepdims=True
            )
            mx.eval(denominator, moment)
        denominator = mx.where(denominator > 0, denominator, 1)
        shift = moment / denominator
        grad = mx.zeros_like(g)
        for first, last, keys, scores in blocks(query):
            values = v[:, :, None, first:last].astype(mx.float32)
            weights = mx.exp(scores - maxima) / denominator
            grad = (
                grad
                + ((weights * (g @ values.swapaxes(-1, -2) - shift)) @ keys) * scale
            )
            mx.eval(grad)
        return grad.reshape(query.shape).astype(query.dtype)

    return forward(q)
