"""Exact vocabulary projection diagnostics on one real long-supervision training shape."""

import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .. import checkpoint
from ..config import config
from ..data.v2 import rows
from ..data.tokenize import batch
from ..io import save
from ..training.optimizers import optimizer


def run():
    c = config("outputs/phase3/configs/p3_direct.yaml")
    # Four distinct real traces. Freeze this one shape for compilation and all timing repeats.
    sample = sorted(rows("validation", "trace")[:60], key=lambda r: r["output_length"])[
        -4:
    ]
    x, p, y = batch(sample)

    def project(h, weight, labels):
        return mx.fast.cross_entropy(h @ weight.T, mx.stop_gradient(labels)).sum()

    compiled = mx.compile(project)

    def objective(method, h, w, labels):
        if method == "full":
            return project(h, w, labels)
        if method == "compiled":
            return compiled(h, w, labels)
        total = 0
        for i in range(0, labels.size, 32):
            chunk_labels = labels[i : i + 32]
            if method == "recomputed":
                # Only hidden states are differentiable in this frozen tied head.
                def chunk_loss(states, targets=chunk_labels):
                    return project(states, w, targets)

                total = total + mx.checkpoint(chunk_loss)(h[i : i + 32])
            else:
                total = total + project(h[i : i + 32], w, chunk_labels)
        return total

    results = {}
    for method in ("full", "chunk32", "recomputed", "compiled"):
        print(f"Testing exact projection: {method}", flush=True)
        model, _, _ = checkpoint.setup(c)
        model.checkpointing = True
        model.train()

        def loss(m, x, p, y):
            hidden = m.hidden(x)
            return (
                objective(
                    method,
                    hidden.reshape(-1, hidden.shape[-1])[p],
                    m.text.model.embed_tokens.weight,
                    y,
                )
                / y.size
            )

        h = model.hidden(x)
        selected = mx.stop_gradient(h.reshape(-1, h.shape[-1])[p])
        weight = model.text.model.embed_tokens.weight
        mx.eval(selected)

        ref, dr = mx.value_and_grad(lambda z, w: objective("full", z, w, y) / y.size)(
            selected, weight
        )
        mx.eval(ref, dr)
        start = perf_counter()
        actual, da = mx.value_and_grad(
            lambda z, w: objective(method, z, w, y) / y.size
        )(selected, weight)
        mx.eval(actual, da, ref, dr)
        setup_seconds = perf_counter() - start
        e = mx.sqrt(
            mx.mean((da.astype(mx.float32) - dr) ** 2)
            / mx.mean(dr.astype(mx.float32) ** 2)
        ).item()
        loss_error = abs(actual.item() - ref.item())
        del selected, h, da, dr, ref, actual
        opt = optimizer(c)
        fn = nn.value_and_grad(model, loss)
        samples = []
        mx.reset_peak_memory()
        for _ in range(5):
            start = perf_counter()
            value, g = fn(model, x, p, y)
            g, norm = optim.clip_grad_norm(g, c["clip"])
            opt.update(model, g)
            mx.eval(model.trainable_parameters(), opt.state, norm, value)
            samples.append(perf_counter() - start)
        results[method] = dict(
            setup_s=setup_seconds,
            warmup_s=samples[0],
            steady_s=samples[1:],
            median_s=median(samples[1:]),
            peak_bytes=mx.get_peak_memory(),
            hidden_grad_relative_rms=e,
            abs_mean_loss_error=loss_error,
            final_training_loss=value.item(),
            final_grad_norm=norm.item(),
        )
        del model, weight, opt, fn, g
        gc.collect()
        mx.clear_cache()
    save(
        "outputs/phase3/projection.json",
        dict(
            ids=[r["id"] for r in sample],
            shape=list(x.shape),
            labels=y.size,
            checkpointing=True,
            k=2,
            chunk=32,
            results=results,
        ),
    )


if __name__ == "__main__":
    run()
