"""Full-history, query-only attention differentiation through native MLX Metal partitions."""

from functools import cache

import mlx.core as mx


@cache
def partition_kernel():
    # Every SIMD group owns one query and one consecutive section of its
    # physical KV head. FP32 online softmax shares the same local normalizer
    # for all moments. No document-sized FP32 copy is required.
    return mx.fast.metal_kernel(
        name="loopqwen_frozen_document_moments",
        input_names=["query", "keys", "values", "upstream", "alpha"],
        output_names=["partial"],
        ensure_row_contiguous=False,
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint part = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
        const uint bh = threadgroup_position_in_grid.z, heads = query_shape[1];
        const uint batch = bh/heads, head = bh%heads, kv = head/(heads/keys_shape[1]);
        const uint length = query_shape[2], n = keys_shape[2];
        const uint first = part * Tile, last = min(n, first + uint(Tile));
        const uint end = Causal ? min(last, n-length+row+1) : last;
        constexpr uint Width = D/32;
        constexpr uint Out = Backward ? 2*D+3 : D+2;
        float q[Width], g[Width], a[Width], b[Width];
        const uint qi = batch*query_strides[0]+head*query_strides[1]+row*query_strides[2];
        const uint gi = batch*upstream_strides[0]+head*upstream_strides[1]+row*upstream_strides[2];
        for (uint j=0;j<Width;j++) {
            const uint c=32*j+lane;
            q[j]=float(query[qi+c*query_strides[3]])*alpha[0];
            if (Backward) g[j]=float(upstream[gi+c*upstream_strides[3]]);
            a[j]=0; b[j]=0;
        }
        float largest=-INFINITY, z=0, mean_dot=0;
        const uint kb=batch*keys_strides[0]+kv*keys_strides[1];
        const uint vb=batch*values_strides[0]+kv*values_strides[1];
        for (uint i=first;i<end;i++) {
            float key[Width], value[Width], s=0, dot=0;
            for (uint j=0;j<Width;j++) {
                const uint c=32*j+lane;
                key[j]=float(keys[kb+i*keys_strides[2]+c*keys_strides[3]]);
                value[j]=float(values[vb+i*values_strides[2]+c*values_strides[3]]);
                s=fma(q[j],key[j],s);
                if (Backward) dot=fma(g[j],value[j],dot);
            }
            const float score=simd_sum(s);
            if (Backward) dot=simd_sum(dot);
            const float m=max(largest,score);
            const float old=exp(largest-m), w=exp(score-m);
            largest=m; z=fma(old,z,w);
            if (Backward) mean_dot=fma(old,mean_dot,w*dot);
            for (uint j=0;j<Width;j++) {
                a[j]=fma(old,a[j],w*(Backward ? key[j] : value[j]));
                if (Backward) b[j]=fma(old,b[j],w*dot*key[j]);
            }
        }
        const uint parts=(n+Tile-1)/Tile;
        const uint base=((bh*length+row)*parts+part)*Out;
        if (lane==0) {
            partial[base]=largest; partial[base+1]=z;
            if (Backward) partial[base+2+2*D]=mean_dot;
        }
        for (uint j=0;j<Width;j++) {
            const uint c=32*j+lane;
            partial[base+2+c]=a[j];
            if (Backward) partial[base+2+D+c]=b[j];
        }
        """,
    )


def document_attention(q, k, v, scale, mask, *, tile=512):
    """Evaluate and differentiate only the queries, with immutable shared source K and V."""
    b, heads, length, dim = q.shape
    if (mask is not None and not isinstance(mask, str)) or mask not in (None, "causal"):
        raise ValueError(
            "The full-document kernel supports an unmasked or causal suffix"
        )
    if (
        dim not in (32, 64, 128, 256)
        or k.shape != v.shape
        or k.shape[0] != b
        or k.shape[-1] != dim
        or heads % k.shape[1]
        or k.shape[2] < length
        or tile < 1
    ):
        raise ValueError("Incompatible full-document attention dimensions")
    parts = (k.shape[2] + tile - 1) // tile
    alpha = mx.array([scale], dtype=mx.float32)

    def moments(query, incoming, backward):
        width = 2 * dim + 3 if backward else dim + 2
        (result,) = partition_kernel()(
            inputs=[query, k, v, incoming, alpha],
            template=[
                ("D", dim),
                ("Tile", tile),
                ("Backward", backward),
                ("Causal", mask == "causal"),
            ],
            grid=(32 * parts, length, b * heads),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, heads, length, parts, width)],
            output_dtypes=[mx.float32],
        )
        maxima = result[..., :1]
        weight = mx.exp(maxima - mx.max(maxima, axis=-2, keepdims=True))
        totals = (result[..., 1:] * weight).sum(axis=-2)
        normalized = totals[..., 1:] / totals[..., :1]
        if not backward:
            return normalized.astype(query.dtype)
        key, mixed = normalized[..., :dim], normalized[..., dim : 2 * dim]
        dot = normalized[..., -1:]
        return (scale * (mixed - dot * key)).astype(query.dtype)

    @mx.custom_function
    def forward(query):
        return moments(mx.stop_gradient(query), mx.stop_gradient(query), False)

    @forward.vjp
    def backward(query, incoming, _result):
        return moments(mx.stop_gradient(query), mx.stop_gradient(incoming), True)

    return forward(q)
