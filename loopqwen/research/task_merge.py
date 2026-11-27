"""Merge aligned, large parameter displacements in the actual shared transformer projections."""

from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

from ..io import digest, read, save


def consensus(changes, density):
    """Use per-source and per-projection trimming, sign election and a disjoint mean."""
    if not 0 < density <= 1 or not changes:
        raise ValueError("A positive density and at least one change are required")
    shape = changes[0].shape
    trimmed = []
    cutoffs = []
    for change in changes:
        if change.shape != shape:
            raise ValueError("All task changes must represent the same physical weight")
        change = change.astype(mx.float32)
        absolute = mx.abs(change).reshape(-1)
        keep = max(1, int(density * absolute.size))
        cutoff = mx.partition(absolute, absolute.size - keep)[absolute.size - keep]
        retained = mx.where(mx.abs(change) >= cutoff, change, 0)
        mx.eval(cutoff, retained)
        trimmed.append(retained)
        cutoffs.append(cutoff.item())
    values = mx.stack(trimmed)
    elected = mx.sign(mx.sum(values, axis=0))
    agree = (values != 0) & (mx.sign(values) == elected)
    count = mx.sum(agree, axis=0)
    merged = mx.sum(mx.where(agree, values, 0), axis=0) / mx.maximum(count, 1)
    metrics = {
        "cutoffs": cutoffs,
        "retained_per_source": mx.sum(
            values != 0, axis=tuple(range(1, values.ndim))
        ).tolist(),
        "opposite_signs": mx.sum(
            mx.any(values > 0, axis=0) & mx.any(values < 0, axis=0)
        ).item(),
        "nonzero": mx.sum(merged != 0).item(),
        "participation": [mx.sum(count == i).item() for i in range(len(changes) + 1)],
        "elements": merged.size,
    }
    return merged, metrics


def effect(tensors, name, scale):
    a, b = (tensors[name + "." + key] for key in ("lora_a", "lora_b"))
    return scale * (b.T.astype(mx.float32) @ a.T.astype(mx.float32))


def compile_consensus(directory, root):
    """Consume raw low-rank tensors and fold the consensus only once into each donor weight."""
    from ..tool_model import bind

    start = perf_counter()
    spec = read(root / "registration.json")
    net = bind(None, Path(directory) / "tool")
    original_folder = Path(spec["checkpoint"]) / "tool"
    original_path = original_folder / "tool-adapters.safetensors"
    assert digest(original_path) == read(original_folder / "tool-model.json")["weights"]
    anchor = mx.load(str(original_path))
    sources = []
    recipe = spec["merged_task_updates"]
    for parent in recipe["parents"]:
        folder = Path("research/results") / parent["id"]
        old = read(folder / "registration.json")
        assert all(
            old[k] == spec[k]
            for k in (
                "donor",
                "prefix_bits",
                "rank",
                "start_layer",
                "checkpoint",
                "instruction_interpolation",
            )
        )
        assert digest(folder / "checksums.json") == parent["archive"]
        source = folder / "tool-adapters.safetensors"
        assert (
            digest(source)
            == read(folder / "checksums.json")[source.name]
            == read(folder / "fit.json")["weights"]
        )
        sources.append(mx.load(str(source)))
    fresh = root / "tool-adapters.safetensors"
    assert digest(fresh) == read(root / "fit.json")["weights"]
    sources.append(mx.load(str(fresh)))
    assert all(v.keys() == anchor.keys() for v in sources)
    tensors, metrics = {}, {}
    layer = net.text.layers[spec["start_layer"]]
    modules = dict(layer.named_modules())
    for key in spec["instruction_interpolation"]["projections"]:
        mod = modules[key]
        assert isinstance(mod, LoRALinear)
        prefix = "text.model.layers." + str(spec["start_layer"]) + "." + key

        original = effect(anchor, prefix, mod.scale)
        changes = [effect(params, prefix, mod.scale) - original for params in sources]
        adjustment, record = consensus(changes, recipe["density"])
        fused = (mod.linear.weight.astype(mx.float32) + original + adjustment).astype(
            mod.linear.weight.dtype
        )
        mx.eval(fused)
        assert mx.isfinite(fused).all().item()
        name = prefix.removeprefix("text.") + ".weight"
        tensors[name] = fused
        metrics[name] = record
    weights = root / "cross-view.safetensors"
    mx.save_safetensors(str(weights), tensors)
    save(
        root / "task-merge.json",
        {
            "original": digest(original_path),
            "parents": recipe["parents"],
            "new_expert": digest(fresh),
            "density": recipe["density"],
            "projections": metrics,
            "compiled": digest(weights),
        },
    )
    save(
        root / "compiled.json",
        {
            "weights": digest(weights),
            "adapter": digest(fresh),
            "registration": digest(root / "registration.json"),
            "projections": sorted(tensors),
            "bytes": weights.stat().st_size,
            "merge": digest(root / "task-merge.json"),
            "peak_bytes": mx.get_peak_memory(),
            "seconds": perf_counter() - start,
        },
    )
