"""Native recurrent decoding with separate full-history and local attention sites."""

from mlx_lm.models.cache import KVCache, RotatingKVCache

from .looped_model import LoopedModel
from .recurrence import Span


class HybridAttention(LoopedModel):
    global_sites = frozenset({(15, 1), (19, 0), (23, 0)})

    def __init__(self, text, window=1024):
        super().__init__(
            text, Span(start=15, length=1, passes=2, mode="stabilized"), alpha=0.35
        )
        self.local_window = window

    def make_cache(self, max_kv_size=None, *, left_padding=None):
        if max_kv_size is not None or left_padding is not None:
            raise ValueError(
                "The validated hybrid layout requires a single full-history request"
            )
        cache = super().make_cache()
        for site, slot in cache.slots.items():
            if isinstance(slot, KVCache) and site not in self.global_sites:
                cache.slots[site] = RotatingKVCache(max_size=self.local_window, keep=0)
        object.__setattr__(self, "last_cache", cache)
        return cache

    def history(self):
        evidence = []
        for (layer, turn), slot in self.last_cache.slots.items():
            if isinstance(slot, (KVCache, RotatingKVCache)):
                evidence.append(
                    dict(
                        layer=layer,
                        turn=turn,
                        offset=slot.offset,
                        retained=slot.state[0].shape[-2],
                        global_attention=(layer, turn) in self.global_sites,
                    )
                )
        return evidence
