"""Full-history inference with original 16-bit activations and immutable KV prefixes on disk."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


from .output_projection import project_output


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
        if prefix["k"].shape[2] != self.length or prefix["v"].shape[2] != self.length:
            raise ValueError("Incomplete staged attention history")
        k, v = self.tail.update_and_fetch(keys, values)
        return mx.concatenate([prefix["k"], k], axis=2), mx.concatenate(
            [prefix["v"], v], axis=2
        )

    def source_values(self):
        values = mx.load(str(self.path))["v"]
        if values.shape[2] != self.length:
            raise ValueError("Incomplete source value history")
        if self.tail.empty():
            return values
        return mx.concatenate([values, self.tail.keys_and_values()[1]], axis=2)

    def physical_length(self):
        return self.length + (
            0 if self.tail.empty() else self.tail.keys_and_values()[0].shape[2]
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

    def __init__(self, model, directory, *, group_attention=False):
        self.model = model
        self.directory = Path(directory)
        self.cache = model.make_cache()
        self.groups = stages(model)
        self.group_attention = group_attention
        if group_attention:
            from .bounded_read import BoundedContentRead
            from .content_read import ContentRead
            from .looped_model import LoopedModel
            from .shared_query import SharedQuery
            from .source_copy import SourceCopyRead

            if type(model) not in (
                LoopedModel,
                SharedQuery,
                ContentRead,
                BoundedContentRead,
                SourceCopyRead,
            ) or any(p for _, p in model.sites):
                raise ValueError(
                    "Grouped staging requires an ordinary or shared-query decoder with one temporal state per layer"
                )
            self.groups = []
            for site in model.sites:
                if not self.groups or not model.text.layers[site[0]].is_linear:
                    self.groups.append([])
                self.groups[-1].append(site)

    def prefill(self, ids, progress=None):
        return project_output(
            self.model, self.prefill_hidden(ids, progress), self.cache
        )

    def prefill_hidden(self, ids, progress=None):
        m = self.model
        self.cache.validate(m, None)
        length = len(ids)
        if hasattr(m, "remember"):
            m.remember(self.cache, mx.array([ids]))
        if self.group_attention:
            # Reserve each site's actual request length only when it becomes active.
            from .reserved_cache import reserve

            reserve(self.cache, length)
        hidden_path = self.directory / "hidden.u16"
        words = np.memmap(
            hidden_path,
            dtype=np.uint16,
            mode="w+",
            shape=(length, m.args.hidden_size),
        )
        dtype = None
        for index, group in enumerate(self.groups):
            self.active_sites = group
            last = index == len(self.groups) - 1
            final = getattr(m, "staged_final", None) if last else None
            if final is not None and group != [(len(m.text.layers) - 1, 0)]:
                raise ValueError("Staged output requires a single final memory owner")
            for first in range(0, length, 128):
                self.consumed_in_group = first
                end = min(length, first + 128)
                if progress is not None:
                    progress(first if index == len(self.groups) - 1 else 0)
                if index == 0:
                    h = m.text.model.embed_tokens(mx.array([ids[first:end]]))
                    dtype = h.dtype
                    if dtype not in (mx.bfloat16, mx.float16):
                        raise ValueError(
                            "Disk-backed prefill requires 16-bit activations"
                        )
                else:
                    # Copy before replacing these same rows in the mapped file.
                    h = mx.array(words[first:end].copy()).view(dtype)[None]
                if final is not None:
                    h = final(h, self.cache.slots[group[0]], emit=end == length)
                    if h is None:
                        continue
                else:
                    h = m._execute(h, group, self.cache)
                mx.eval(h)
                if not last:
                    if h.dtype != dtype:
                        raise ValueError("A staged layer changed its activation dtype")
                    words[first:end] = np.asarray(h[0].view(mx.uint16))
            for site in group:
                slot = self.cache.slots[site]
                if isinstance(slot, KVCache):
                    if slot.offset != length:
                        raise ValueError("Incomplete attention history")
                    k, v = slot.keys_and_values()
                    path = self.directory / f"kv-{site[0]}-{site[1]}.safetensors"
                    mx.save_safetensors(str(path), {"k": k, "v": v})
                    assert k.shape[2] == v.shape[2] == length
                    self.cache.slots[site] = PrefixOnDisk(path, k.shape[2])
                    del slot, k, v
            if self.group_attention:
                mx.clear_cache()
        self.scratch_bytes = sum(
            file.stat().st_size for file in self.directory.iterdir()
        )
        del words
        hidden_path.unlink()
        return m.text.model.norm(h[:, -1:])

    def fork(self, length):
        """Fork a completed staged prefix while sharing only its immutable files."""
        from mlx.utils import tree_map
        from mlx_lm.models.cache import ArraysCache
        from .tool_prefill import copy_state

        self.cache.validate(self.model, None)
        child = OffloadedPrefill(
            self.model, self.directory, group_attention=self.group_attention
        )
        for site, slot in self.cache.slots.items():
            if (
                isinstance(slot, PrefixOnDisk)
                and slot.offset == length
                and slot.tail.empty()
            ):
                child.cache.slots[site] = PrefixOnDisk(slot.path, length)
            elif isinstance(slot, ArraysCache):
                child.cache.slots[site].state = tree_map(copy_state, slot.state)
            else:
                raise ValueError(
                    "A document must have complete immutable prefix states before branching"
                )
        if hasattr(self.cache, "source_ids"):
            child.cache.source_ids = self.cache.source_ids[...]
        child.scratch_bytes = self.scratch_bytes
        return child

    def decode(self, ids):
        return project_output(self.model, self.decode_hidden(ids), self.cache)

    def decode_hidden(self, ids, *, emit=True):
        m = self.model
        self.cache.validate(m, None)
        if hasattr(m, "remember"):
            m.remember(self.cache, ids)
        h = m.text.model.embed_tokens(ids)
        for index, group in enumerate(self.groups):
            final = (
                getattr(m, "staged_final", None)
                if index == len(self.groups) - 1
                else None
            )
            h = (
                final(h, self.cache.slots[group[0]], emit=emit)
                if final
                else m._execute(h, group, self.cache)
            )
            if h is None:
                return None
            # Release this stage's loaded prefix before the next stage reads its own.
            mx.eval(h)
        return m.text.model.norm(h)


class DocumentPrefix:
    """Reuse an unchanged token prefix for independent requests to the same frozen model.

    The caller owns an empty scratch directory and keeps it alive for every request.
    """

    def __init__(self, model, ids, directory, progress=None):
        if model.training or not ids:
            raise ValueError("A nonempty prefix requires a fixed evaluation-mode model")
        directory = Path(directory)
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError("Document staging requires an empty private directory")
        self.ids = tuple(ids)
        self.disk = OffloadedPrefill(model, directory, group_attention=True)
        self.disk.prefill_hidden(self.ids, progress)

    def branch(self, model, opening):
        if (
            model is not self.disk.model
            or tuple(opening[: len(self.ids)]) != self.ids
            or len(opening) <= len(self.ids)
        ):
            raise ValueError(
                "A request must extend this exact document with the same model"
            )
        return self.disk.fork(len(self.ids))

    def complete_opening(self, disk, opening, progress=None):
        for start in range(len(self.ids), len(opening), 128):
            end = min(start + 128, len(opening))
            hidden = disk.decode_hidden(
                mx.array([opening[start:end]]), emit=end == len(opening)
            )
            if progress is not None:
                progress(start)
        return project_output(disk.model, hidden[:, -1:], disk.cache)
