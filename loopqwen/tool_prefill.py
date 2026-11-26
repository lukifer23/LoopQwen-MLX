"""Reuse the exact common prefix of a single multi-turn tool request."""

import mlx.core as mx
from mlx.utils import tree_map


def copy_state(value):
    # Separate Python array handles retain value semantics across indexed KV writes.
    return value[...] if isinstance(value, mx.array) else value


class ToolPrefix:
    def __init__(self):
        self.tokens = ()
        self.snapshot = None
        self.boundary = None
        self.owner = None
        self.reused = 0

    def identity(self, model):
        return (
            model._generation,
            model.span,
            id(getattr(model, "alpha_logit", None)),
            model.alpha_override,
        )

    def begin(self, model, ids, block, cache):
        self.reused = 0
        if self.owner is None:
            self.owner = self.identity(model)
            # Keep aligned blocks only, leaving the last input token to produce the first output.
            self.boundary = min(4096 // block, max(0, (len(ids) - 1) // block)) * block
        if (
            self.snapshot is None
            or self.owner != self.identity(model)
            or len(ids) <= len(self.tokens)
            or tuple(ids[: len(self.tokens)]) != self.tokens
        ):
            return 0
        for site, state in self.snapshot.items():
            cache.slots[site].state = tree_map(copy_state, state)
        self.reused = len(self.tokens)
        return self.reused

    def capture(self, model, ids, offset, cache):
        if (
            self.snapshot is None
            and self.owner == self.identity(model)
            and offset == self.boundary
        ):
            self.snapshot = {
                site: tree_map(copy_state, slot.state)
                for site, slot in cache.slots.items()
            }
            self.tokens = tuple(ids[:offset])
