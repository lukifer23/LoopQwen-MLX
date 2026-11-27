"""Share a frozen full-document prefix across independent supervised continuations."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.cache import ArraysCache

from .offloaded_prefill import OffloadedPrefill, PrefixOnDisk
from .shared_query import SharedQuery, attend, memory
from .tool_prefill import copy_state


class DocumentMemory:
    """The caller owns the private directory until all branches have finished."""

    def __init__(self, model, ids, directory, progress=None):
        if type(model) is not SharedQuery or model.training or not ids:
            raise ValueError(
                "A nonempty document prefix and fixed evaluation-mode shared-query decoder are required"
            )
        self.model = model
        self.ids = tuple(ids)
        self.disk = OffloadedPrefill(model, Path(directory), group_attention=True)
        self.disk.prefill_hidden(ids, progress)
        self.disk.cache.validate(model, None)
        self.final = model.sites[-1]
        if self.disk.groups[-1] != [self.final]:
            raise ValueError("The last group needs one attention owner")
        for slot in self.disk.cache.slots.values():
            if isinstance(slot, PrefixOnDisk):
                if slot.offset != len(ids) or not slot.tail.empty():
                    raise ValueError("The immutable prefix has an unexpected length")
            elif not isinstance(slot, ArraysCache):
                raise ValueError("Unsupported prefix cache type")

    def branch(self, opening, suffix):
        """Return causal states for every target; the source KV file is stored only once."""
        self.disk.cache.validate(self.model, None)
        if (
            tuple(opening[: len(self.ids)]) != self.ids
            or len(opening) <= len(self.ids)
            or not suffix
        ):
            raise ValueError(
                "A branch must extend the exact original token prefix with an opening and a target"
            )
        net = self.model
        # Independent Python array handles and slot objects prevent a sibling branch
        # from mutating the convolution, delta state, RoPE offset or new KV tail.
        cache = net.make_cache()
        for site, slot in self.disk.cache.slots.items():
            if isinstance(slot, PrefixOnDisk):
                cache.slots[site] = PrefixOnDisk(slot.path, slot.length)
            else:
                cache.slots[site].state = tree_map(copy_state, slot.state)
        tail = list(opening[len(self.ids) :]) + list(suffix[:-1])
        first = len(opening) - len(self.ids) - 1
        selected = []
        ordinary = []
        # Materialize each temporal group before loading the next immutable KV file.
        # Each group runs all new tokens, preserving causal order inside the branch.
        h = net.text.model.embed_tokens(mx.array([tail]))
        for group in self.disk.groups[:-1]:
            pieces = []
            for off in range(0, len(tail), 128):
                pieces.append(net._execute(h[:, off : off + 128], group, cache))
                mx.eval(pieces[-1])
            h = mx.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
            mx.eval(h)
        layer = net.text.layers[-1]
        slot = cache.slots[self.final]
        for off in range(0, len(tail), 128):
            block = h[:, off : off + 128]
            k, v, pos, mask = memory(layer, block, slot, None)
            start = max(0, first - off)
            if start < block.shape[1]:
                pre = block[:, start:]
                if isinstance(mask, mx.array):
                    mask = mask[..., start:, :]
                elif pre.shape[1] == 1:
                    mask = None
                y = attend(layer, pre, k, v, pos + start, mask)
                mx.eval(y)
                selected.append(pre)
                ordinary.append(y)
            else:
                mx.eval(slot.tail.state)
        k, v = slot.tail.keys_and_values()
        assert k.shape[2] == v.shape[2] == len(tail)
        pre = mx.concatenate(selected, axis=1)
        h = mx.concatenate(ordinary, axis=1)
        assert pre.shape == h.shape and h.shape[1] == len(suffix)
        return {
            "pre": pre,
            "ordinary": h,
            "tail_keys": k,
            "tail_values": v,
            "labels": mx.array(suffix),
            "offset": len(opening) - 1,
            "prefix_file": slot.path,
            "prefix_length": slot.length,
        }
