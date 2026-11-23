import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear


def attach(model, policy="block", rank=8, alpha=16, projections=None):
    model.freeze()
    if model.span.has_gate:
        model.unfreeze(keys=["alpha_logit"], recurse=False)
    n = len(model.text.layers)
    start, end = model.span.start, model.span.start + model.span.length
    ranges = {
        "block": range(start, end),
        "boundary": range(max(0, start - 1), min(n, end + 1)),
        "nearby": range(max(0, start - 2), min(n, end + 2)),
        "last8": range(n - 8, n),
        "all": range(n),
    }
    if policy not in ranges or rank < 1 or alpha <= 0:
        raise ValueError("Invalid adapter configuration")
    names = []
    for i in ranges[policy]:
        layer = model.text.layers[i]
        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            parent, key = name.rsplit(".", 1)
            if projections is not None and key not in projections:
                continue
            adapter = LoRALinear.from_base(
                module, r=rank, scale=alpha / rank, dropout=0
            )
            setattr(getattr(layer, parent), key, adapter)
            names.append(f"{i}.{name}")
    if not names:
        raise ValueError("No matching LoRA modules")
    return names
