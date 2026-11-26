"""Frozen ordinary first pass and a causal, input-dependent second application."""

import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import create_attention_mask

from .looped_model import LoopedModel
from .operand_planner import adapted_attention_layer
from .recurrence import Span


def build(spec):
    from .model_loader import load_base
    from .tool_model import quantize_prefix

    text, tok, _ = load_base()
    quantize_prefix(text, len(text.layers) - 1, 8)
    mx.random.seed(spec["fit_seed"])
    model = SelectiveRefinement(text, spec["rank"], spec["initial_gate"])
    model.eval()
    return model, tok


def load_checkpoint(directory):
    from .io import digest, read
    from .research.raw_distillation import check_weights

    directory = Path(directory)
    inventory = read(directory / "checksums.json")
    for name in ("registration.json", "fit.json", "refinement.safetensors"):
        if digest(directory / name) != inventory[name]:
            raise ValueError(
                "Selective checkpoint file differs from its inventory: " + name
            )
    spec = read(directory / "registration.json")
    meta = read(directory / "fit.json")
    if (
        meta["weights"] != inventory["refinement.safetensors"]
        or meta["updates"] != spec["updates"]
    ):
        raise ValueError("Selective training receipt differs from the supplied weights")
    model, tok = build(spec)
    check_weights(model, mx.load(str(directory / "refinement.safetensors")))
    mx.eval(model.parameters())
    return model, tok


class SelectiveRefinement(LoopedModel):
    def __init__(self, text, rank, initial_gate):
        last = len(text.layers) - 1
        super().__init__(text, Span(start=last, length=1, passes=2))
        self.freeze()
        self.refinement = adapted_attention_layer(self, last, rank)
        self.gate = nn.Linear(self.args.hidden_size, 1)
        self.gate.weight = mx.zeros_like(self.gate.weight)
        self.gate.bias = mx.full((1,), math.log(initial_gate / (1 - initial_gate)))

    def ordinary(self, prefix, slot=None, window_size=None):
        return self.text.layers[-1](
            prefix,
            mask=create_attention_mask(prefix, slot, window_size=window_size),
            cache=slot,
        )

    def refine(self, ordinary, slot=None, window_size=None, observer=None):
        changed = self.refinement(
            ordinary,
            mask=create_attention_mask(ordinary, slot, window_size=window_size),
            cache=slot,
        )
        gate = mx.sigmoid(self.gate(self.text.model.norm(ordinary).astype(mx.float32)))
        hidden = ordinary + gate.astype(ordinary.dtype) * (changed - ordinary)
        if observer is not None:
            observer("exit", 1, hidden)
        return self.text.model.norm(hidden), gate[..., 0]

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        return self.hidden_and_gate(ids, cache, window_size, observer)[0]

    def hidden_and_gate(self, ids, cache=None, window_size=None, observer=None):
        """Return the causal interpolation used in this same decoder call."""
        if cache is not None:
            cache.validate(self, window_size)
        last = self.span.start
        original = self._execute(
            self.text.model.embed_tokens(ids), self.sites[:-2], cache, window_size
        )
        if observer is not None:
            observer("entry", 0, original)
        head = self.ordinary(
            original, None if cache is None else cache.slots[last, 0], window_size
        )
        if observer is not None:
            observer("exit", 0, head)
            observer("entry", 1, head)
        return self.refine(
            head, None if cache is None else cache.slots[last, 1], window_size, observer
        )
