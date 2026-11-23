from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


class SiteCache:
    """One convolution+delta state or KV pair for each (layer, pass) site."""
    def __init__(self, model, max_kv_size=None):
        self.sites = model.sites
        if max_kv_size is not None and max_kv_size < 2:
            raise ValueError("KV window must contain at least two tokens")
        self.slots = {}
        for i,p in self.sites:
            if model.text.layers[i].is_linear:
                slot = ArraysCache(2)
            elif max_kv_size:
                slot = RotatingKVCache(max_size=max_kv_size, keep=0)
            else:
                slot = KVCache()
            self.slots[i,p] = slot
