"""Restartable full-history prefixes for fixed, ordinary-head Qwen span models."""

from pathlib import Path
from time import monotonic

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache

from ..identity import fingerprint
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..offloaded_prefill import DocumentPrefix, OffloadedPrefill, PrefixOnDisk
from ..reserved_cache import reserve


class SavedDocument(DocumentPrefix):
    """Persist each completed physical group before releasing its input activations.

    The caller holds the GPU lease. Each commit references only completed files.
    An interrupted group replays from its last immutable input. KV files and
    delta/conv states are distinct for every physical visit, even for tied layers.
    """

    def __init__(self, model, ids, directory, identity, progress):
        if type(model) is not LoopedModel or model.training or not ids:
            raise ValueError(
                "A fixed native Qwen span and a nonempty prefix are required"
            )
        self.ids = tuple(ids)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        disk = self.disk = OffloadedPrefill(model, directory, group_attention=True)
        length = len(ids)
        manifest = directory / "prefix.json"
        key = fingerprint([identity, model.sites, vars(model.span), ids, 128])
        state = (
            read(manifest)
            if manifest.exists()
            else dict(identity=key, groups=[], seconds=0)
        )
        if state["identity"] != key:
            raise ValueError(
                "The prefix belongs to a different model, input or schedule"
            )
        reserve(disk.cache, length)
        for entry in state["groups"]:
            for item in entry["states"]:
                site = tuple(item["site"])
                path = directory / item["file"]
                if digest(path) != item["sha256"]:
                    raise ValueError("A persisted temporal state changed")
                if item["kind"] == "kv":
                    disk.cache.slots[site] = PrefixOnDisk(path, length)
                else:
                    arrays = mx.load(str(path))
                    disk.cache.slots[site].state = [
                        arrays[str(i)] for i in range(len(arrays))
                    ]
        dtype = model.text.model.embed_tokens.weight.dtype
        if dtype not in (mx.bfloat16, mx.float16):
            raise ValueError("Exact 16-bit prefill storage requires a 16-bit model")
        shape = (length, model.args.hidden_size)
        for index in range(len(state["groups"]), len(disk.groups)):
            tick = monotonic()
            previous = None
            if index:
                item = state["groups"][-1]
                path = directory / item["hidden"]
                if digest(path) != item["hidden_sha256"]:
                    raise ValueError("Incomplete saved group input")
                previous = np.memmap(path, mode="r", dtype=np.uint16, shape=shape)
            last = index + 1 == len(disk.groups)
            output = directory / f"hidden-{index}.u16"
            mapped = (
                None
                if last
                else np.memmap(output, mode="w+", dtype=np.uint16, shape=shape)
            )
            group = disk.groups[index]
            for first in range(0, length, 128):
                progress(first)
                end = min(first + 128, length)
                h = (
                    model.text.model.embed_tokens(mx.array([ids[first:end]]))
                    if previous is None
                    else mx.array(previous[first:end].copy()).view(dtype)[None]
                )
                h = model._execute(h, group, disk.cache)
                mx.eval(h)
                if mapped is not None:
                    mapped[first:end] = np.asarray(h[0].view(mx.uint16))
            if mapped is not None:
                mapped.flush()
            del mapped, previous
            entries = []
            for site in group:
                slot = disk.cache.slots[site]
                path = directory / ("site-%d-%d.safetensors" % site)
                if isinstance(slot, ArraysCache):
                    values = {str(i): s for i, s in enumerate(slot.state)}
                    kind = "hybrid"
                else:
                    k, v = slot.keys_and_values()
                    assert k.shape[2] == v.shape[2] == slot.offset == length
                    values = {"k": k, "v": v}
                    kind = "kv"
                mx.save_safetensors(str(path), values)
                entries.append(
                    dict(
                        site=site,
                        file=path.name,
                        sha256=digest(path),
                        kind=kind,
                        bytes=path.stat().st_size,
                    )
                )
                if kind == "kv":
                    disk.cache.slots[site] = PrefixOnDisk(path, length)
                    del k, v
                del slot, values
            event = dict(states=entries, seconds=monotonic() - tick)
            if not last:
                event.update(hidden=output.name, hidden_sha256=digest(output))
            state["groups"].append(event)
            state["seconds"] += event["seconds"]
            save(manifest, state)
            if index:
                (directory / state["groups"][index - 1]["hidden"]).unlink(
                    missing_ok=True
                )
            mx.clear_cache()
        disk.scratch_bytes = sum(p.stat().st_size for p in directory.iterdir())
        self.receipt = state
