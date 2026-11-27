"""A two-visit content reader with a per-token bound on the residual write."""

import mlx.core as mx
from mlx import nn

from .content_read import ContentRead, ContentStep
from .shared_query import SharedQuery

FRACTION = 1 / 16


class NormalizedWrite(nn.Linear):
    def __call__(self, recalled):
        return super().__call__(mx.fast.rms_norm(recalled, None, 1e-6))


class BoundedStep(ContentStep):
    def __init__(self, width, rank, gate_rank, heads, dimension):
        super().__init__(width, rank, gate_rank, heads, dimension)
        write = NormalizedWrite(heads * dimension, width, bias=False)
        write.weight = self.write.weight
        self.write = write

    def recall(self, state, values, mask, *, differentiate, observe_content=None):
        x = state.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        unit = x / rms
        # Reuse the complete causal content-attention operator. Its FP32
        # residual on a unit state is the dimensionless projected read.
        read = (
            super().recall(
                unit,
                values,
                mask,
                differentiate=differentiate,
                observe_content=observe_content,
            )
            - unit
        )
        return (x + FRACTION * rms * mx.tanh(read)).astype(state.dtype)


class BoundedContentRead(ContentRead):
    def __init__(self, text, *, revisits, rank, gate_rank):
        SharedQuery.__init__(
            self, text, revisits=revisits, rank=rank, gate_rank=gate_rank
        )
        a = text.layers[-1].self_attn
        self.step = BoundedStep(
            text.args.hidden_size, rank, gate_rank, a.num_key_value_heads, a.head_dim
        )
