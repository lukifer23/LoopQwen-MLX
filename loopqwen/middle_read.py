"""Requery the middle global layer's immutable evidence before the ordinary suffix."""

import mlx.core as mx
from mlx import nn

from .recurrence import Span
from .shared_query import QueryStep, attend, attention_update, memory

LAYER = 15


class MiddleRead(nn.Module):
    is_linear = False

    def __init__(self, block, width, reads):
        super().__init__()
        if block.is_linear or reads not in (1, 2):
            raise ValueError(
                "Middle evidence reads require one global layer and one or two reads"
            )
        self.block = block
        self.step = QueryStep(width, rank=8, gate_rank=4)
        self._reads = reads

    def __call__(self, x, mask=None, cache=None):
        # Only the native layer input contributes a key/value at this position.
        k, v, offset, _ = memory(self.block, x, cache, None, mask_needed=False)
        state = attend(self.block, x, k, v, offset, mask)
        for _ in range(self._reads):
            update = attention_update(
                self.block, state, k, v, offset, mask, step=self.step
            )
            state = self.step(state, update)
        return state


def install(model, reads, seed):
    if len(model.text.layers) != 24 or model.span.start != 12 or model.span.length != 4:
        raise ValueError("Expected the registered middle-span parent")
    if isinstance(model.text.layers[LAYER], MiddleRead):
        raise TypeError("The middle read is already installed")
    model.freeze()
    mx.random.seed(seed)
    model.span = Span(12, 4, 1)
    layer = MiddleRead(model.text.layers[LAYER], model.args.hidden_size, reads)
    model.text.model.layers[LAYER] = layer
    model.checkpointing = False
    model.eval()
    model._generation = object()
    return layer


def set_reads(model, count):
    layer = model.text.layers[LAYER]
    if not isinstance(layer, MiddleRead) or count not in (1, 2):
        raise ValueError("Expected a trained middle read and a registered read count")
    if layer._reads != count:
        layer._reads = count
        model._generation = object()
