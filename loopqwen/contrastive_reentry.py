"""A recurrent readout whose trainable corrections require a physical layer update."""

import math

import mlx.core as mx
from mlx import nn

from .conditional_reentry import Correction


class IncrementReadout(nn.Module):
    def __init__(self, width, rank, gate_rank, passes):
        super().__init__()
        if passes not in (1, 2):
            raise ValueError("Expected one or two physical visits")
        self.passes = passes
        bound = 1 / math.sqrt(width)
        self.down = mx.random.uniform(-bound, bound, (4, rank, width))
        self.up = nn.Linear(4 * rank, width, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)
        self.context = Correction(width, gate_rank)
        self.bias = mx.zeros((width,))

    def __call__(self, features):
        states = features.astype(mx.float32)
        entry, final = states[..., 0, :], states[..., -1, :]
        ordinary = entry if self.passes == 2 else final
        scale = mx.sqrt(mx.mean(mx.square(ordinary), -1, keepdims=True) + 1e-6)
        gate = mx.tanh(self.bias + self.context(ordinary / scale))
        increments = (states[..., 1:, :] - states[..., :-1, :]) / scale[..., None, :]
        # Batch the four independent down projections and sum their up projections
        # in one matrix multiply. This representation is also used during fitting.
        compressed = (increments[..., None, :] @ self.down.swapaxes(-1, -2)).squeeze(-2)
        hidden = nn.silu(compressed).reshape(*ordinary.shape[:-1], -1)
        update = self.up(hidden) * scale
        return (ordinary + gate * (final - entry) + (1 + gate) * update).astype(
            features.dtype
        )
