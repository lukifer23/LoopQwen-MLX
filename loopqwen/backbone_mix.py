"""Materialize a full-rank task vector in the 24 shared decoder layers."""

import mlx.core as mx
from mlx.utils import tree_flatten


def layer_weights(model):
    return {
        name: value
        for name, value in tree_flatten(model.text.parameters())
        if name.startswith("model.layers.")
        and not name.endswith((".lora_a", ".lora_b"))
    }


def apply(model, original, donor, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("A backbone mixture needs a fraction in (0, 1]")
    active = layer_weights(model)
    if original.keys() != active.keys():
        raise ValueError("Original decoder tensors changed")
    references = layer_weights(donor)
    replacements = []
    for name, base in original.items():
        sibling = references[name.replace(".linear.weight", ".weight")]
        if base.shape != sibling.shape or base.dtype != sibling.dtype:
            raise ValueError(f"Incompatible donor tensor {name}")
        if fraction == 1:
            result = sibling
        else:
            x = base.astype(mx.float32)
            result = (x + fraction * (sibling.astype(mx.float32) - x)).astype(
                base.dtype
            )
        mx.eval(result)
        replacements.append((name, result))
    model.text.load_weights(replacements, strict=False)
    return len(replacements)


def restore(model, spec):
    # The source pin, lexical IDs, tensor file and text configuration are verified here.
    from .research.projected_distillation import tutor

    sibling, _ = tutor({"teacher": spec["donor"]})
    count = apply(model, layer_weights(model), sibling, spec["fraction"])
    if count != spec["tensors"]:
        raise ValueError("Incomplete decoder mixture")
