"""Reuse the exact common prefix of a single multi-turn tool request."""

import mlx.core as mx
from mlx.utils import tree_map


def copy_state(value):
    # Separate Python array handles retain value semantics across indexed KV writes.
    return value[...] if isinstance(value, mx.array) else value


class ToolPrefix:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
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

    def prepare(self, model, ids, block):
        if self.owner is None or (self.boundary == 0 and self.snapshot is None):
            self.owner = self.identity(model)
            # Keep aligned blocks only, leaving the last input token to produce the first output.
            limit = len(ids) - 1
            if self.max_tokens is not None:
                limit = min(limit, self.max_tokens)
            self.boundary = (limit // block) * block

    def begin(self, model, ids, block, cache):
        self.prepare(model, ids, block)
        self.reused = 0
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


class ConversationPrefill:
    """Restore the longest exact saved state and retain one private conversation prefix."""

    def __init__(self, conversation, within_request):
        self.prefixes = (
            (conversation, within_request) if within_request else (conversation,)
        )
        self.reused = 0

    def begin(self, model, ids, block, cache):
        for prefix in self.prefixes:
            prefix.prepare(model, ids, block)
        for prefix in sorted(self.prefixes, key=lambda p: len(p.tokens), reverse=True):
            self.reused = prefix.begin(model, ids, block, cache)
            if self.reused:
                break
        return self.reused

    def capture(self, model, ids, offset, cache):
        for prefix in self.prefixes:
            prefix.capture(model, ids, offset, cache)
