"""Coalesce the two frozen quantized SwiGLU inputs without dequantizing or changing rows."""

from copy import copy

import mlx.core as mx
from mlx import nn
from mlx_lm.models.activations import swiglu


class PairedInputMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        gate, up = mlp.gate_proj, mlp.up_proj
        if not isinstance(gate, nn.QuantizedLinear) or not isinstance(
            up, nn.QuantizedLinear
        ):
            raise TypeError("Paired input requires quantized gate and up projections")
        if (gate.group_size, gate.bits, gate.mode) != (up.group_size, up.bits, up.mode):
            raise ValueError("Quantization formats differ")
        keys = set(gate.parameters())
        if (
            keys != set(up.parameters())
            or not {"weight", "scales"} <= keys
            or keys - {"weight", "scales", "biases", "bias"}
        ):
            raise ValueError("Projection tensor sets differ")
        if any(
            gate[k].shape != up[k].shape or gate[k].dtype != up[k].dtype for k in keys
        ):
            raise ValueError("Projection tensor shapes or dtypes differ")
        projection = copy(gate)
        for k in keys:
            projection[k] = mx.concatenate((gate[k], up[k]), axis=0)
        self.gate_up = projection
        self.down_proj = mlp.down_proj
        mx.eval(self.gate_up.parameters())

    def __call__(self, x):
        gate, up = mx.split(self.gate_up(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def pair_frozen_prefix(model):
    if model.training:
        raise ValueError("Paired projections require an inference model")
    layers = model.text.layers[: model.span.start]
    for layer in layers:
        layer.mlp = PairedInputMLP(layer.mlp)
    return len(layers)
