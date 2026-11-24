"""Matched native timing and gradient checks on real categorical prompts."""

import statistics
from time import perf_counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from ..training.mechanism import requests, model_for
from ..training import pointer5 as kernel
from ..data import pointer5
from ..data.pointer6 import ordered
from ..evaluation.mechanism import load
from ..io import save, read, digest
from ..identity import fingerprint


def timed(fn):
    mx.synchronize()
    begin = perf_counter()
    x = fn()
    mx.eval(x)
    return x, perf_counter() - begin


def bench(arm):
    m, t, st, w = load(arm, 480)
    path = arm.output / "benchmark.json"
    ident = fingerprint([st["identity"], w, digest(Path(__file__))])
    if path.exists():
        if read(path)["identity"] != ident:
            raise ValueError("Incompatible timing benchmark")
        return
    group = [ordered(r, "random0", 61091) for r in pointer5.rows("development")[:8]]
    x, last = kernel.inputs(requests(group, 1, arm, t, {}))

    def exits(independent):
        logits = (
            [kernel.outputs(m, x, last, k, "recurrent")[-1] for k in range(1, 7)]
            if independent
            else kernel.outputs(m, x, last, 6, "recurrent")
        )
        return mx.softmax(mx.stack(logits, axis=1).astype(mx.float32), axis=-1)

    profile = {}
    for k in [True, False]:
        timed(lambda: exits(k))
    for k in [True, False, False, True, True, False]:
        mx.reset_peak_memory()
        p, dt = timed(lambda: exits(k))
        profile.setdefault(str(k), []).append(
            dict(seconds=dt, peak=mx.get_peak_memory())
        )
    old = exits(True)
    new = exits(False)
    err = mx.max(mx.abs(old - new)).item()
    agreement = mx.sum(mx.argmax(old, axis=-1) == mx.argmax(new, axis=-1)).item()
    if err > 1e-5 or agreement != 48:
        raise ValueError("Shared unroll failed parity")

    # Include build cost for two consumers of the same eight tokenized logical batches.
    # Prefix reuse is legal only for frozen embeddings/layers before the LoRA span.
    names = dict(tree_flatten(m.trainable_parameters()))
    if any(
        "embed_tokens" in name or any(f"layers.{i}." in name for i in range(10))
        for name in names
    ):
        raise ValueError("Trainable prefix is ineligible for reuse")
    rows = pointer5.rows("train")[:64]
    batches = []
    for i in range(0, len(rows), 8):
        g = [ordered(r, arm.rendering, arm.seed) for r in rows[i : i + 8]]
        a, b = kernel.inputs(requests(g, 1, arm, t, {}))
        batches.append((a, b, mx.array([r["states"][:3] for r in g])))

    def loss(model, a, b, y, cached):
        h = a if cached else model.prefix_hidden(a)
        losses = []
        for j in range(3):
            h = model.shared_block(h)
            losses.append(
                nn.losses.cross_entropy(
                    kernel.readout(model, h, b), y[:, j], reduction="mean"
                )
            )
        return sum(losses) / 3

    m.train()
    grad = nn.value_and_grad(m, loss)
    cache, creation = timed(lambda: [m.prefix_hidden(a) for a, b, y in batches])
    raw = grad(m, *batches[0], False)
    ref = grad(m, cache[0], *batches[0][1:], True)

    def vec(tree):
        return mx.concatenate([v.reshape(-1) for _, v in tree_flatten(tree)])

    g1, g2 = vec(raw[1]), vec(ref[1])
    relative = (mx.linalg.norm(g1 - g2) / mx.linalg.norm(g1)).item()
    if abs(raw[0].item() - ref[0].item()) > 1e-5 or relative > 0.004:
        raise ValueError("Prefix gradient check failed")
    memory = sum(h.nbytes for h in cache)
    initial = m.trainable_parameters()
    mx.eval(initial)
    results = []
    for cached in (False, True, True, False):
        m.update(initial)
        opt = kernel.adam()
        times = []
        for i, (a, b, y) in enumerate(batches):

            def step():
                value, g = grad(m, cache[i] if cached else a, b, y, cached)
                import mlx.optimizers as optim

                g, _ = optim.clip_grad_norm(g, 1.0)
                opt.update(m, g)
                return [value, m.trainable_parameters(), opt.state]

            _, dt = timed(step)
            times.append(dt)
        results.append(dict(cached=cached, update_seconds=sum(times), samples=times))
    # Measure actual disk write and read once, including materialization.
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        f = Path(folder) / "prefix.safetensors"
        then = perf_counter()
        mx.save_safetensors(str(f), {str(j): h for j, h in enumerate(cache)})
        write = perf_counter() - then
        _, fetch = timed(lambda: mx.load(str(f)))
    save(
        path,
        dict(
            identity=ident,
            weights=w,
            independent_vs_single=profile,
            prob_max_abs=err,
            argmax_agree=agreement,
            prefix=dict(
                batches=len(batches),
                cache_bytes=memory,
                creation_seconds=creation,
                disk_write_seconds=write,
                disk_read_seconds=fetch,
                grad_relative_rms=relative,
                updates=results,
                key=fingerprint(
                    [
                        st["provenance"]["assets"],
                        "bf16",
                        "right-pad",
                        [a.tolist() for a, b, y in batches],
                    ]
                ),
            ),
        ),
    )
