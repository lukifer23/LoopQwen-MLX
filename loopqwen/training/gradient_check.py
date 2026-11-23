import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from ..io import save
from ..looped_model import LoopedModel
from ..model_loader import load_base, count
from .adapters import attach


def gradient_check():
    base, tok, _ = load_base()
    mx.random.seed(41)
    model = LoopedModel(base)
    names = attach(model)
    model.train()
    ids = mx.array(
        [
            tok.encode(
                "When one apple and two more apples are put together, there are three apples in the basket."
            )
        ]
    )

    def loss(m):
        return mx.fast.cross_entropy(m(ids[:, :-1]), ids[:, 1:]).mean()

    fn = nn.value_and_grad(model, loss)
    opt = optim.AdamW(1e-4)
    rows = []
    for step in range(2):
        before = model.text.layers[10].mlp.up_proj.lora_b
        value, grad = fn(model)
        norms = {k: mx.linalg.norm(v) for k, v in tree_flatten(grad)}
        mx.eval(norms)
        norms = {k: v.item() for k, v in norms.items()}
        if not all(v > 0 for k, v in norms.items() if step or k.endswith("lora_b")):
            raise AssertionError(norms)
        opt.update(model, grad)
        mx.eval(model.trainable_parameters(), opt.state)
        delta = mx.abs(model.text.layers[10].mlp.up_proj.lora_b - before).max().item()
        assert delta > 0
        rows.append(
            dict(loss=value.item(), min_gradient=min(norms.values()), max_update=delta)
        )
    result = dict(adapters=names, trainable=count(model, True), rows=rows)
    save("outputs/gradient_check.json", result)
    return result
