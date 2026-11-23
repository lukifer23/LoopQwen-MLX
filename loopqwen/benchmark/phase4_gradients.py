"""Compare the optimized gradient to independently evaluated exits on real weights."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint
from ..config import config
from ..data.tokenize import batch
from ..data.v2 import rows
from ..io import save
from ..training.losses import exit_objective, loss_sum
from ..training.optimizers import optimizer


def rms(a, b):
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(a), 1e-12)).item()


def run(bf16=False):
    cfg = config("outputs/phase4/configs/p4_profile_360_all.yaml")
    cfg["target"] = "boundary"
    model, _, _ = checkpoint.setup(cfg)
    model.checkpointing = True
    if not bf16:
        model.set_dtype(mx.float32)
    model.train()
    samples = rows("validation", "answer")[:2]
    init = model.trainable_parameters()

    # Isolate the objective and batching from native BF16 rounding.
    def independent(m, x, p, y, w):
        orig = m.span
        m.span = replace(orig, passes=1)
        l1 = loss_sum(m, x, p, y)
        m.span = orig
        l2 = loss_sum(m, x, p, y)
        return w * l1 + (1 - w) * l2

    def joint(m, x, p, y, w):
        return exit_objective(m, x, p, y, 1.0, w)[0]

    def result(fn, groups, weight):
        model.update(init)
        total = 0
        grads = None
        labels = 0
        value_grad = nn.value_and_grad(model, fn)
        for group in groups:
            x, p, y = batch(group)
            loss, g = value_grad(model, x, p, y, weight)
            total += loss
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            labels += y.size
            mx.eval(total, grads)
        grads = tree_map(lambda g: g / labels, grads)
        old = dict(tree_flatten(init))
        flat = dict(tree_flatten(grads))
        opt = optimizer(cfg)
        opt.update(model, grads)
        updated = dict(tree_flatten(model.trainable_parameters()))
        delta = mx.concatenate([(v - old[k]).reshape(-1) for k, v in updated.items()])
        x, p, y = batch(samples)
        output = model.project(model.hidden(x).reshape(-1, 1024)[p])
        mx.eval(delta, output)
        return total.item() / labels, flat, delta, output

    metrics = {}

    def compare(ref, actual):
        l, g, d, p = ref
        v, h, e, q = actual
        return dict(
            loss_abs=abs(l - v),
            grad_rms=rms(
                mx.concatenate([g[k].flatten() for k in g]),
                mx.concatenate([h[k].flatten() for k in g]),
            ),
            suffix_grad_norm=mx.linalg.norm(
                mx.concatenate(
                    [a.flatten() for k, a in h.items() if ".layers.16." in k]
                )
            ).item(),
            gate_ref=g["alpha_logit"].item(),
            gate_actual=h["alpha_logit"].item(),
            update_rms=rms(d, e),
            logits_rms=rms(p, q),
        )

    for w in (0.0, 0.25, 0.5, 1.0):
        ref = result(independent, [samples], w)
        shared = result(joint, [samples], w)
        metrics[str(w)] = compare(ref, shared)
        metrics[f"split-{w}"] = compare(ref, result(joint, [[r] for r in samples], w))
    destination = "gradient-bf16.json" if bf16 else "gradient-exits.json"
    save(Path("outputs/phase4") / destination, metrics)
    for rec in metrics.values():
        if (
            rec["loss_abs"] > (0.08 if bf16 else 2e-4)
            or rec["grad_rms"] > (0.15 if bf16 else 0.002)
            or rec["logits_rms"] > (0.04 if bf16 else 0.002)
            or rec["update_rms"] > (0.35 if bf16 else 0.04)
            or rec["suffix_grad_norm"] == 0
            or abs(rec["gate_actual"] - rec["gate_ref"]) > (0.02 if bf16 else 0.001)
        ):
            raise ValueError(f"Gradient audit failed: {rec}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args().bf16)
