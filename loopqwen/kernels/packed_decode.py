"""Native Metal single-query attention over the pinned decoder's affine 8-bit KV rows."""

from functools import cache

import mlx.core as mx


@cache
def partition_kernel():
    # Four SIMD groups share a KV head. Each SIMD group owns one query head;
    # each lane accumulates eight contiguous dimensions of its output vector.
    return mx.fast.metal_kernel(
        name="loopqwen_packed_decode_partition",
        input_names=[
            "query",
            "kcode",
            "kscale",
            "kbias",
            "vcode",
            "vscale",
            "vbias",
            "bias",
        ],
        output_names=["partial"],
        ensure_row_contiguous=False,
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint kv_head = threadgroup_position_in_grid.y;
        const uint head = 4 * kv_head + simdgroup_index_in_threadgroup;
        const uint part = threadgroup_position_in_grid.x;
        const uint count = kcode_shape[2];
        const uint first = part * Block;
        const uint stop = min(count, first + uint(Block));
        const uint group = lane / 8;
        float q[8];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint j = 0; j < 8; ++j) {
            q[j] = float(query[head * query_strides[1] + lane * 8 + j]) * 0.0625f;
        }

        float maximum = -INFINITY;
        float total = 0;
        for (uint row = first; row < stop; ++row) {
            const uint ki = kv_head * kcode_strides[1] + row * kcode_strides[2] + 2 * lane;
            const uint vi = kv_head * vcode_strides[1] + row * vcode_strides[2] + 2 * lane;
            const uint ks = kv_head * kscale_strides[1] + row * kscale_strides[2] + group;
            const uint kb = kv_head * kbias_strides[1] + row * kbias_strides[2] + group;
            const uint vs = kv_head * vscale_strides[1] + row * vscale_strides[2] + group;
            const uint vb = kv_head * vbias_strides[1] + row * vbias_strides[2] + group;
            const float sk = float(kscale[ks]), bk = float(kbias[kb]);
            const uint2 kw = uint2(kcode[ki], kcode[ki + 1]);
            float product = 0;
            for (uint j = 0; j < 8; ++j) {
                float code = float((kw[j / 4] >> (8 * (j % 4))) & 255);
                product += q[j] * fma(code, sk, bk);
            }
            // Match the BF16 score and additive bias used by the reference path.
            float score = float(bfloat(simd_sum(product)));
            if (HasBias) score = float(bfloat(score + float(bias[row])));
            if (score == -INFINITY) continue;
            float next_max = max(maximum, score);
            float old_weight = total == 0 ? 0 : exp(maximum - next_max);
            float new_weight = exp(score - next_max);
            const float sv = float(vscale[vs]), bv = float(vbias[vb]);
            const uint2 vw = uint2(vcode[vi], vcode[vi + 1]);
            for (uint j = 0; j < 8; ++j) {
                float code = float((vw[j / 4] >> (8 * (j % 4))) & 255);
                acc[j] = fma(new_weight, fma(code, sv, bv), old_weight * acc[j]);
            }
            total = fma(old_weight, total, new_weight);
            maximum = next_max;
        }
        const uint partitions = (count + Block - 1) / Block;
        const uint base = (head * partitions + part) * 258;
        for (uint j = 0; j < 8; ++j) partial[base + lane * 8 + j] = acc[j];
        if (lane == 0) {
            partial[base + 256] = maximum;
            partial[base + 257] = total;
        }
        """,
    )


@mx.compile
def merge_partitions(partial):
    maxima = partial[:, :, 256:257]
    change = mx.exp(maxima - mx.max(maxima, axis=1, keepdims=True))
    normalizer = (change * partial[:, :, 257:258]).sum(axis=1)
    merged = (change * partial[:, :, :256]).sum(axis=1) / normalizer
    return merged.astype(mx.bfloat16)[None, :, None, :]


def attention(query, key, value, scale, mask=None, *, factored=False):
    """One resident request with eight query heads and two native KV heads."""
    if query.shape != (1, 8, 1, 256) or query.dtype != mx.bfloat16 or scale != 0.0625:
        raise ValueError(
            "This kernel requires the pinned single-token Qwen BF16 query layout"
        )
    length = key[0].shape[2]
    for code, scales, biases in (key, value):
        if (
            code.shape != (1, 2, length, 64)
            or code.dtype != mx.uint32
            or scales.shape != biases.shape
            or scales.shape != (1, 2, length, 4)
            or scales.dtype != biases.dtype
            or scales.dtype != mx.bfloat16
        ):
            raise ValueError(
                "This kernel requires eight-bit codes and four BF16 affine groups per row"
            )
    use_bias = mask is not None and not isinstance(mask, str)
    if isinstance(mask, str) and mask != "causal":
        raise ValueError("Unsupported decoder attention mask")
    if use_bias:
        if mask.size != length:
            raise ValueError("The decoder bias must cover the entire consumed history")
        bias = (
            mx.where(mask, 0.0, mx.finfo(mx.bfloat16).min)
            if mask.dtype == mx.bool_
            else mask
        )
        bias = bias.reshape(length).astype(mx.bfloat16)
    else:
        bias = mx.zeros((1,), mx.bfloat16)
    block = 256
    parts = (length + block - 1) // block
    if factored:
        from .factored_decode import partition_kernel as factored_kernel

        kernel = factored_kernel()
    else:
        kernel = partition_kernel()
    partial = kernel(
        inputs=[query, *key, *value, bias],
        template=[("Block", block), ("HasBias", use_bias)],
        grid=(128 * parts, 2, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(8, parts, 258)],
        output_dtypes=[mx.float32],
    )[0]
    return merge_partitions(partial)
