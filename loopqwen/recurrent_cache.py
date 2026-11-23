from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, RotatingKVCache


class SiteCache:
    """One convolution+delta state or KV pair for each (layer, pass) site."""

    def __init__(self, model, max_kv_size=None, *, left_padding=None):
        self.sites = model.sites
        if max_kv_size is not None and max_kv_size < 2:
            raise ValueError("KV window must contain at least two tokens")
        if left_padding is not None and (
            max_kv_size is not None or not left_padding or min(left_padding) < 0
        ):
            raise ValueError(
                "Left-padded batches require nonnegative padding and full KV history"
            )
        self.slots = {}
        for i, p in self.sites:
            if model.text.layers[i].is_linear:
                slot = ArraysCache(2, left_padding=left_padding)
            elif left_padding is not None:
                slot = BatchKVCache(left_padding)
            elif max_kv_size:
                slot = RotatingKVCache(max_size=max_kv_size, keep=0)
            else:
                slot = KVCache()
            self.slots[i, p] = slot

    def finish_left_prefill(self):
        for slot in self.slots.values():
            if isinstance(slot, ArraysCache):
                slot.finalize()
