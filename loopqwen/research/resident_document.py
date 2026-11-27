"""Materialize each immutable full-precision prefix once in a private decode cache."""

import mlx.core as mx

from ..offloaded_prefill import DocumentPrefix, PrefixOnDisk
from ..reserved_cache import ReservedKVCache


class ResidentDocument(DocumentPrefix):
    """Keep the staged builder and fork independent exact histories for each turn."""

    def __init__(self, source, extra_tokens, guard):
        if extra_tokens < 1:
            raise ValueError("A decode branch requires output headroom")
        self.ids, self.disk = source.ids, source.disk
        self.extra_tokens = extra_tokens
        self.guard = guard

    def branch(self, model, opening):
        disk = super().branch(model, opening)
        disk.materialized_kv_bytes = 0
        for site, prefix in disk.cache.slots.items():
            if not isinstance(prefix, PrefixOnDisk):
                continue
            self.guard()
            kv = mx.load(str(prefix.path))
            live = ReservedKVCache(len(opening) + self.extra_tokens)
            if kv["k"].shape[2] != prefix.length or kv["v"].shape[2] != prefix.length:
                raise ValueError(
                    "Incomplete prefix history while materializing a decode branch"
                )
            k, v = live.update_and_fetch(kv["k"], kv["v"])
            mx.eval(k, v)
            disk.cache.slots[site] = live
            disk.materialized_kv_bytes += prefix.prefix_bytes
            del kv, k, v
        return disk


class StagedVisitDocument(DocumentPrefix):
    """Use a separate exact staged history and a materialization boundary per visit."""

    def __init__(self, source):
        self.ids, self.disk = source.ids, source.disk

    def branch(self, model, opening):
        reader = super().branch(model, opening)
        reader.materialize_visits = True
        return reader
