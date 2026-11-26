"""Causal page selection for interior prefill on the pinned Qwen attention layout."""

import mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention


class PagePolicy:
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

    def page_indices(self, queries):
        length = queries.shape[2]
        prev = self.offset - length
        pages = prev // self.block
        # Only the first query of the chunk can influence page choice. It is
        # visible to every subsequent query and has no access to later inputs.
        b, heads, _, dim = queries.shape
        kv_heads = self.summaries.shape[1]
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
        return indices

    def report(self):
        return {
            "offset": self.offset,
            "retained": self.physical_length(),
            "sparse_blocks": self.sparse_blocks,
            "dense_queries": self.dense_queries,
            "attended": self.attended,
            "possible": self.possible,
        }


class PageCache(PagePolicy, KVCache):
    def physical_length(self):
        return self.keys_and_values()[0].shape[2]

    def select(self, queries, keys, values):
        indices = self.page_indices(queries)
        return mx.take(keys, indices, axis=2), mx.take(values, indices, axis=2)


class PackedPageCache(PagePolicy, QuantizedKVCache):
    """Affine 8-bit key/value groups. Gather pages before restoring BF16."""

    def physical_length(self):
        arrays = [a for bundle in self.keys_and_values() for a in bundle]
        lengths = {a.shape[2] for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                "Packed KV data and affine factors cover different positions"
            )
        return lengths.pop()

    def restore(self, bundles, queries=None):
        indices = self.page_indices(queries) if queries is not None else None

        def dequant(bundle):
            if indices is not None:
                bundle = [mx.take(a, indices, axis=2) for a in bundle]
            return mx.dequantize(*bundle, group_size=self.group_size, bits=self.bits)

        return tuple(dequant(bundle) for bundle in bundles)

    def report(self):
        bundles = self.keys_and_values()
        return dict(
            super().report(),
            bits=self.bits,
            group_size=self.group_size,
            code_bytes=sum(b[0].nbytes for b in bundles),
            affine_bytes=sum(a.nbytes for b in bundles for a in b[1:]),
        )


class DirectPackedPageCache(PackedPageCache):
    """Reserve packed storage once and read long dense tails without restoring full KV."""

    tile_queries = 8

    def __init__(self, prompt_length, capacity, *, fused_decode=False):
        super().__init__(prompt_length)
        self.fused_decode = fused_decode
        self.fused_queries = 0
        if capacity < prompt_length:
            raise ValueError("Packed cache capacity must cover the prompt")
        self.capacity = capacity
        self.direct_queries = self.direct_tiles = self.max_tile_queries = 0

    def update_and_fetch(self, keys, values):
        if self.offset + keys.shape[2] > self.capacity:
            raise ValueError("Packed cache request exceeds its reserved capacity")
        if self.keys is None:

            def reserve(array):
                b, h, _, width = array.shape
                shape = (b, h, self.capacity)
                return (
                    mx.zeros((*shape, width // 4), mx.uint32),
                    mx.zeros((*shape, width // self.group_size), array.dtype),
                    mx.zeros((*shape, width // self.group_size), array.dtype),
                )

            self.keys, self.values = reserve(keys), reserve(values)
        return super().update_and_fetch(keys, values)

    def dense_attention(self, q, k, v, scale, mask):
        length = q.shape[2]
        if self.fused_decode and length == 1 and self.offset >= 32768:
            from .kernels.packed_decode import attention

            # Single-query partitions feed the next native layer directly. The
            # per-tile barrier below bounds the multiple resident prefill tiles.
            value = attention(q, k, v, scale, mask)
            self.fused_queries += 1
            self.direct_tiles += 1
            self.direct_queries += 1
            self.max_tile_queries = max(self.max_tile_queries, 1)
            return value
        previous = self.offset - length
        chunks = []
        for lo in range(0, length, self.tile_queries):
            hi = min(length, lo + self.tile_queries)
            tile_mask = mask
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError("Unsupported attention mask")
                tile_mask = (
                    mx.arange(previous + lo, previous + hi)[:, None]
                    >= mx.arange(self.offset)[None, :]
                )
            elif mask is not None and mask.shape[-2] != 1:
                tile_mask = mask[..., lo:hi, :]
            value = quantized_scaled_dot_product_attention(
                q[:, :, lo:hi],
                k,
                v,
                scale,
                tile_mask,
                group_size=self.group_size,
                bits=self.bits,
            )
            # Keep the live score workspace bounded even under lazy evaluation.
            mx.eval(value)
            chunks.append(value)
            self.direct_tiles += 1
            self.max_tile_queries = max(self.max_tile_queries, hi - lo)
        self.direct_queries += length
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)

    def report(self):
        return dict(
            super().report(),
            capacity=self.keys[0].shape[2],
            **({"fused_queries": self.fused_queries} if self.fused_decode else {}),
            direct_queries=self.direct_queries,
            direct_tiles=self.direct_tiles,
            max_tile_queries=self.max_tile_queries,
        )


class PageAttention(Qwen3NextAttention):
    def __call__(self, x, mask=None, cache=None):
        packed = isinstance(cache, PackedPageCache)
        sparse = isinstance(cache, PagePolicy) and cache.sparse(x.shape[1])
        if not (packed or sparse):
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
        if (
            isinstance(cache, DirectPackedPageCache)
            and not sparse
            and cache.offset > 4096
        ):
            h = cache.dense_attention(q, k, v, self.scale, mask)
        else:
            if packed:
                k, v = cache.restore((k, v), q if sparse else None)
            else:
                k, v = cache.select(q, k, v)
            # MLX aligns the causal mask with the trailing query rows. The entire
            # current chunk follows the selected past, in original position order.
            h = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal" if sparse else mask
            )
        h = h.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.o_proj(h * mx.sigmoid(gate.reshape(b, n, -1)))


def enable_page_attention(model):
    """Allow the native attention sites to consume a request's page cache."""
    for layer in model.text.layers:
        if not layer.is_linear:
            attn = layer.self_attn
            if type(attn) is PageAttention:
                continue
            if type(attn) is not Qwen3NextAttention:
                raise TypeError("Unrecognized Qwen attention layout")
            object.__setattr__(attn, "__class__", PageAttention)


def install(cache, length, *, packed=False, direct_capacity=None, fused_decode=False):
    slots = {}
    for site, slot in cache.slots.items():
        if type(slot) is KVCache:
            if slot.offset:
                raise ValueError("Page prefill requires a fresh cache")
            if direct_capacity is not None:
                slot = DirectPackedPageCache(
                    length, direct_capacity, fused_decode=fused_decode
                )
            else:
                slot = PackedPageCache(length) if packed else PageCache(length)
            slots[site] = cache.slots[site] = slot
    return slots


def register_cache(model, route, length, max_tokens, block, cache):
    """Use packed pages only on the long register path of an opted-in export."""
    if (
        route == "register"
        and length >= 16384
        and block == 128
        and getattr(model, "packed_register_prefill", False)
    ):
        enable_page_attention(model)
        return install(
            cache,
            length,
            direct_capacity=length + max_tokens - 1,
            fused_decode=getattr(model, "fused_packed_decode", False),
        )
    return {}


def attach(model, length, *, packed=False, direct_capacity=None):
    """Instrument a single comparison arm using the original study interface."""
    original = model.make_cache
    enable_page_attention(model)
    slots = {}

    def make_cache(*args, **kwargs):
        cache = original(*args, **kwargs)
        slots.update(
            install(cache, length, packed=packed, direct_capacity=direct_capacity)
        )
        return cache

    object.__setattr__(model, "make_cache", make_cache)
    return slots
