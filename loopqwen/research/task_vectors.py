"""Fit bounded-rank additive projections of the post-training update in native MLX."""

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

from .. import checkpoint
from ..io import read


def orthonormal(z):
    # The only eigensystem is on the narrow sample dimension.
    eig, vectors = mx.linalg.eigh(z.T @ z, stream=mx.cpu)
    inv = mx.rsqrt(mx.maximum(eig, mx.maximum(eig[-1] * 1e-9, 1e-24)))
    return (z @ vectors) * inv


def factors(delta, rank, power, strength):
    """Economical randomized range approximation. Only the small Gram needs decomposition."""
    out, inp = delta.shape
    r = min(rank, out, inp)
    width = min(r + 8, out, inp)
    if width == out:
        q = mx.eye(out)
    else:
        omega = mx.random.normal((inp, width))
        q = orthonormal(delta @ omega)
        for _ in range(power):
            z = orthonormal(delta.T @ q)
            q = orthonormal(delta @ z)
    compressed = q.T @ delta
    vals, vecs = mx.linalg.eigh(compressed @ compressed.T, stream=mx.cpu)
    vals, vecs = vals[-r:], vecs[:, -r:]
    # Multiply the right factor by singular values directly, which avoids division by small values.
    a = compressed.T @ vecs
    b = (q @ vecs).T
    norm = mx.sum(delta * delta)
    energy = mx.sum(mx.maximum(vals, 0)) / mx.maximum(norm, 1e-20)
    if r < rank:
        a = mx.pad(a, [(0, 0), (0, rank - r)])
        b = mx.pad(b, [(0, rank - r), (0, 0)])
    return a, b * strength, energy


def initialize(s, root, budget, load_teacher):
    from pathlib import Path

    from ..io import save

    parent = Path(s["parent"])
    dep = read(parent / "state.json")["spec"]["deployment"]
    parent_rank = dep["config"]["rank"]
    parent_scale = dep["config"]["lora_alpha"] / parent_rank
    rank = s["rank"] + parent_rank
    dep = dict(
        dep,
        config=dict(
            dep["config"],
            target="all",
            rank=rank,
            lora_alpha=rank,
            gradient_checkpointing=True,
            seed=s["seed"],
        ),
    )
    model, t, _ = checkpoint.setup(dep["config"])
    model.alpha_override = dep.get("alpha_override")
    old = mx.load(str(parent / "adapters.safetensors"))
    teacher, _ = load_teacher(s)
    report = []
    for i, (layer, other) in enumerate(
        zip(model.text.layers, teacher.text.layers, strict=True)
    ):
        refs = dict(other.named_modules())
        for k, module in list(layer.named_modules()):
            if not isinstance(module, LoRALinear):
                continue
            budget.check()
            path = f"text.model.layers.{i}.{k}"
            delta = refs[k].weight.astype(mx.float32) - module.linear.weight.astype(
                mx.float32
            )
            a, b, energy = factors(
                delta, s["rank"], s["power_iterations"], s["import_strength"]
            )
            if path + ".lora_a" in old:
                base_a = old[path + ".lora_a"]
                base_b = parent_scale * old[path + ".lora_b"]
            else:
                base_a = module.lora_a[:, :parent_rank]
                base_b = mx.zeros_like(module.lora_b[:parent_rank])
            module.lora_a = mx.concatenate([base_a, a], axis=1)
            module.lora_b = mx.concatenate([base_b, b], axis=0)
            mx.eval(module.lora_a, module.lora_b, energy)
            report.append(
                {
                    "projection": path,
                    "shape": list(delta.shape),
                    "fraction": energy.item(),
                }
            )
    model.alpha_logit = old["alpha_logit"]
    mx.eval(model.parameters())
    save(root / "projection.json", report)
    return model, t, dep
