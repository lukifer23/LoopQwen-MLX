"""Full-history inference with BF16 intermediate rows and immutable KV prefixes on disk."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


class PrefixOnDisk(KVCache):
    """RoPE uses the complete offset. Only new decode keys accumulate in memory."""

    def __init__(self, path, length):
        self.path = path
        self.length = length
        self.tail = KVCache()

    @property
    def offset(self):
        return self.length + self.tail.offset

    def update_and_fetch(self, keys, values):
        prefix = mx.load(str(self.path))
        k, v = self.tail.update_and_fetch(keys, values)
        return mx.concatenate([prefix["k"], k], axis=2), mx.concatenate(
            [prefix["v"], v], axis=2
        )


def stages(model):
    """Keep both ends of each residual or anchor in one call to the deployed block."""
    span = model.span
    shared = range(span.start, span.start + span.length)
    groups = []
    for site in model.sites:
        layer, repeat = site
        if layer in shared and span.mode == "input_anchor":
            key = ("anchored block",)
        elif layer in shared and repeat and span.mode == "stabilized":
            key = ("residual block", repeat)
        else:
            key = site
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(site)
    return [group for _, group in groups]


class OffloadedPrefill:
    """Single-request workspace. The caller owns the temporary directory lifetime."""

    def __init__(self, model, directory):
        self.model = model
        self.directory = Path(directory)
        self.cache = model.make_cache()
        self.groups = stages(model)

    def prefill(self, ids, progress=None):
        m = self.model
        self.cache.validate(m, None)
        length = len(ids)
        words = np.memmap(
            self.directory / "hidden.bf16",
            dtype=np.uint16,
            mode="w+",
            shape=(length, m.args.hidden_size),
        )
        for index, group in enumerate(self.groups):
            for first in range(0, length, 128):
                end = min(length, first + 128)
                if progress is not None:
                    progress(first if index == len(self.groups) - 1 else 0)
                if index == 0:
                    h = m.text.model.embed_tokens(mx.array([ids[first:end]]))
                    if h.dtype != mx.bfloat16:
                        raise ValueError(
                            "Disk-backed prefill requires BF16 model activations"
                        )
                else:
                    # Copy before replacing these same rows in the mapped file.
                    h = mx.array(words[first:end].copy()).view(mx.bfloat16)[None]
                h = m._execute(h, group, self.cache)
                mx.eval(h)
                if index != len(self.groups) - 1:
                    words[first:end] = np.asarray(h[0].view(mx.uint16))
            for site in group:
                slot = self.cache.slots[site]
                if isinstance(slot, KVCache):
                    if slot.offset != length:
                        raise ValueError("Incomplete attention history")
                    k, v = slot.keys_and_values()
                    path = self.directory / f"kv-{site[0]}-{site[1]}.safetensors"
                    mx.save_safetensors(str(path), {"k": k, "v": v})
                    self.cache.slots[site] = PrefixOnDisk(path, length)
                    del slot, k, v
        del words
        (self.directory / "hidden.bf16").unlink()
        return m.project(m.text.model.norm(h[:, -1:]))

    def decode(self, ids):
        m = self.model
        self.cache.validate(m, None)
        h = m.text.model.embed_tokens(ids)
        for group in self.groups:
            h = m._execute(h, group, self.cache)
            # Release this stage's loaded prefix before the next stage reads its own.
            mx.eval(h)
        return m.project(m.text.model.norm(h))
