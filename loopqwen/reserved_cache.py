"""Reserve full-history KV storage for a known request length on native MLX."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class ReservedKVCache(KVCache):
    def __init__(self, capacity):
        super().__init__()
        if type(capacity) is not int or capacity < 1:
            raise ValueError("KV capacity must be a positive integer")
        self.capacity = capacity

    def update_and_fetch(self, keys, values):
        if self.offset + keys.shape[2] > self.capacity:
            raise ValueError("Full-history request exceeded its reserved KV capacity")
        if self.keys is None:
            size = (*keys.shape[:2], self.capacity)
            self.keys = mx.zeros((*size, keys.shape[-1]), keys.dtype)
            self.values = mx.zeros((*size, values.shape[-1]), values.dtype)
        return super().update_and_fetch(keys, values)


def reserve(cache, capacity):
    for site, slot in cache.slots.items():
        if type(slot) is KVCache:
            if not slot.empty():
                raise ValueError("Reserve full-history KV before prefill")
            cache.slots[site] = ReservedKVCache(capacity)
