"""A tied recurrent read of Qwen's contextual values using a learned content query."""

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import scaled_dot_product_attention

from .shared_query import QueryStep, SharedQuery, differentiable_attention


class ContentStep(QueryStep):
    def __init__(self, width, rank, gate_rank, heads, dimension):
        super().__init__(width, rank, gate_rank)
        self.heads = heads
        self.dimension = dimension
        # Map the updated recurrent state into the existing value space. The zero
        # output projection exactly preserves the fitted parent before training.
        self.address = nn.Linear(width, heads * dimension, bias=False)
        self.write = nn.Linear(heads * dimension, width, bias=False)
        self.write.weight = mx.zeros_like(self.write.weight)

    def recall(self, state, values, mask, *, differentiate, observe_content=None):
        batch, length, width = state.shape
        x = state.astype(mx.float32)
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        q = self.address(x * norm).reshape(batch, length, self.heads, self.dimension)
        q = q.transpose(0, 2, 1, 3).astype(values.dtype)
        if observe_content is not None:
            observe_content(q)
        # The original contextual values carry the token and its causal context.
        # The content query does not rotate with distance from the output.
        if differentiate:
            result = differentiable_attention(
                q, values, values, self.dimension**-0.5, mask
            )
        else:
            result = scaled_dot_product_attention(
                q, values, values, cache=None, scale=self.dimension**-0.5, mask=mask
            )
        z = result.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return (x + self.write(z.astype(mx.float32))).astype(state.dtype)


class ContentRead(SharedQuery):
    def __init__(self, text, *, revisits, rank, gate_rank):
        super().__init__(text, revisits=revisits, rank=rank, gate_rank=gate_rank)
        a = text.layers[-1].self_attn
        self.step = ContentStep(
            text.args.hidden_size, rank, gate_rank, a.num_key_value_heads, a.head_dim
        )

    def refine(
        self,
        pre,
        ordinary,
        k,
        v,
        pos,
        mask,
        *,
        step=None,
        observer=None,
        differentiate=False,
        observe_query=None,
        observe_content=None,
        visits=None,
    ):
        step = self.step if step is None else step
        count = self.revisits if visits is None else visits
        if count not in ((1, 2) if self.revisits else (0,)):
            raise ValueError("Invalid recurrent content-read intervention")
        state = ordinary
        for i in range(max(1, count)):
            state = super().refine(
                pre,
                state,
                k,
                v,
                pos,
                mask,
                step=step,
                differentiate=differentiate,
                visits=1 if count else 0,
                observe_query=observe_query if i + 1 == max(1, count) else None,
            )
            state = step.recall(
                state,
                v,
                mask,
                differentiate=differentiate,
                observe_content=observe_content if i + 1 == max(1, count) else None,
            )
            if observer is not None and count:
                observer("exit", i + 1, state)
        return state
