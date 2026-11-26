"""Partitioned affine-quantized attention with a shared softmax scale for eight rows."""

from functools import cache

import mlx.core as mx


@cache
def partition_kernel():
    return mx.fast.metal_kernel(
        name="loopqwen_factored_decode_partition",
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
        const uint kv = threadgroup_position_in_grid.y;
        const uint head = 4 * kv + simdgroup_index_in_threadgroup;
        const uint part = threadgroup_position_in_grid.x;
        const uint count = kcode_shape[2];
        const uint first = part * Block, stop = min(count, first + uint(Block));
        const uint group = lane / 8;
        float q[8], qsum = 0;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint j = 0; j < 8; ++j) {
            q[j] = float(query[head * query_strides[1] + lane * 8 + j]) * 0.0625f;
            qsum += q[j];
        }
        float maximum = -INFINITY, total = 0, bias_acc = 0;
        for (uint start = first; start < stop; start += 8) {
            const uint rows = min(8u, stop - start);
            float scores[8], next_max = maximum;
            for (uint r = 0; r < rows; ++r) {
                const uint row = start + r;
                const uint i = kv * kcode_strides[1] + row * kcode_strides[2] + 2 * lane;
                const uint2 word = uint2(kcode[i], kcode[i+1]);
                float dot = 0;
                for (uint j = 0; j < 8; ++j)
                    dot = fma(q[j], float((word[j/4] >> (8*(j%4))) & 255), dot);
                const float scale = float(kscale[kv*kscale_strides[1] + row*kscale_strides[2] + group]);
                const float zero = float(kbias[kv*kbias_strides[1] + row*kbias_strides[2] + group]);
                float score = float(bfloat(simd_sum(fma(scale, dot, zero*qsum))));
                if (HasBias) score = float(bfloat(score + float(bias[row])));
                scores[r] = score;
                next_max = max(next_max, score);
            }
            if (next_max == -INFINITY) continue;
            if (next_max > maximum) {
                const float scale = exp(maximum - next_max);
                total *= scale;
                bias_acc *= scale;
                for (uint j = 0; j < 8; ++j) acc[j] *= scale;
                maximum = next_max;
            }
            for (uint r = 0; r < rows; ++r) {
                const uint row = start + r;
                const float weight = exp(scores[r] - maximum);
                total += weight;
                const uint i = kv * vcode_strides[1] + row * vcode_strides[2] + 2 * lane;
                const uint2 word = uint2(vcode[i], vcode[i+1]);
                const float scale = weight * float(vscale[kv*vscale_strides[1] + row*vscale_strides[2] + group]);
                const float zero = float(vbias[kv*vbias_strides[1] + row*vbias_strides[2] + group]);
                bias_acc = fma(weight, zero, bias_acc);
                for (uint j = 0; j < 8; ++j)
                    acc[j] = fma(scale, float((word[j/4] >> (8*(j%4))) & 255), acc[j]);
            }
        }
        const uint parts = (count + Block - 1) / Block;
        const uint base = (head * parts + part) * 258;
        for (uint j = 0; j < 8; ++j) partial[base + lane*8 + j] = acc[j] + bias_acc;
        if (lane == 0) {
            partial[base + 256] = maximum;
            partial[base + 257] = total;
        }
        """,
    )
