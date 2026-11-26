"""Causal page selection for interior prefill on the pinned Qwen attention layout."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention


class PageCache(KVCache):
    """Store every KV row. Keep a mean key per completed prefill page."""

    def __init__(self, prompt_length, block=128, retrieved=16, recent=15, suffix=512):
        super().__init__()
        self.prompt_length = prompt_length
        self.block, self.retrieved, self.recent = block, retrieved, recent
        self.suffix = suffix
        self.summaries = None
        self.sparse_blocks = self.dense_queries = self.attended = self.possible = 0

    def sparse(self, length):
        return (
            length == self.block
            and self.offset % self.block == 0
            and self.offset // self.block > 1 + self.recent + self.retrieved
            and self.offset + length <= self.prompt_length - self.suffix
        )

    def update_and_fetch(self, keys, values):
        length = keys.shape[2]
        end = self.offset + length
        # Counts include only causally visible key/query pairs.
        pairs = self.offset * length + length * (length + 1) // 2
        self.possible += pairs
        self.attended += pairs
        self.dense_queries += length
        if length == self.block and end <= self.prompt_length - self.suffix:
            page = keys.mean(2, keepdims=True)
            self.summaries = (
                page
                if self.summaries is None
                else mx.concatenate([self.summaries, page], 2)
            )
        return super().update_and_fetch(keys, values)

    def select(self, queries, keys, values):
        length = queries.shape[2]
        prev = self.offset - length
        pages = prev // self.block
        # Only the first query of the chunk can influence page choice. It is
        # visible to every subsequent query and has no access to later inputs.
        b, heads, _, dim = queries.shape
        kv_heads = keys.shape[1]
        first = queries[:, :, 0].reshape(b, kv_heads, heads // kv_heads, dim)
        candidates = self.summaries[:, :, 1 : pages - self.recent]
        relevance = (first @ candidates.swapaxes(-1, -2)).max(axis=(1, 2))[0]
        selected = mx.sort(mx.argsort(relevance)[-self.retrieved :] + 1)
        keep = mx.concatenate(
            [
                mx.zeros((1,), mx.int32),
                selected,
                mx.arange(pages - self.recent, pages + 1),
            ]
        )
        indices = (keep[:, None] * self.block + mx.arange(self.block)).reshape(-1)
        self.sparse_blocks += 1
        self.dense_queries -= length
        self.attended -= length * (self.offset - indices.size)
        return mx.take(keys, indices, axis=2), mx.take(values, indices, axis=2)

    def report(self):
        return {
            "offset": self.offset,
            "retained": self.keys_and_values()[0].shape[2],
            "sparse_blocks": self.sparse_blocks,
            "dense_queries": self.dense_queries,
            "attended": self.attended,
            "possible": self.possible,
        }


class PageAttention(Qwen3NextAttention):
    def __call__(self, x, mask=None, cache=None):
        if not isinstance(cache, PageCache) or not cache.sparse(x.shape[1]):
            return super().__call__(x, mask=mask, cache=cache)
        b, n, _ = x.shape
        if b != 1:
            raise ValueError("Page prefill expects a single unpadded request")
        queries, gate = mx.split(
            self.q_proj(x).reshape(b, n, self.num_attention_heads, -1), 2, axis=-1
        )
        q = self.q_norm(queries).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(x).reshape(b, n, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(x)
            .reshape(b, n, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        q, k = self.rope(q, offset=cache.offset), self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        k, v = cache.select(q, k, v)
        # MLX aligns the causal mask with the trailing query rows. The entire
        # current chunk follows the selected past, in original position order.
        h = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal"
        )
        h = h.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.o_proj(h * mx.sigmoid(gate.reshape(b, n, -1)))


def attach(model, length):
    """Apply to one research model instance. The reader's final masks stay dense."""
    original = model.make_cache
    slots = {}
    for layer in model.text.layers:
        if not layer.is_linear:
            attn = layer.self_attn
            if type(attn) is not Qwen3NextAttention:
                raise TypeError("Unrecognized Qwen attention layout")
            object.__setattr__(attn, "__class__", PageAttention)

    def make_cache(*args, **kwargs):
        cache = original(*args, **kwargs)
        for site, slot in cache.slots.items():
            if type(slot) is KVCache:
                slots[site] = cache.slots[site] = PageCache(length)
        return cache

    object.__setattr__(model, "make_cache", make_cache)
    return slots
