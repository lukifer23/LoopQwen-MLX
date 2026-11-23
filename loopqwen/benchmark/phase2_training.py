"""Synchronized end-to-end LoRA updates and real repeated GDN derivative diagnostics."""

import gc
import statistics
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import psutil
from mlx.utils import tree_flatten, tree_map

from ..checkpoint import setup
from ..config import config
from ..data.tokenize import batch
from ..data.v2 import rows
from ..io import save
from ..kernels.gdn_backend import select
from ..training.losses import loss_sum
from ..training.optimizers import optimizer


def relative(a, b):
    return (
        mx.linalg.norm((a - b).astype(mx.float32))
        / mx.maximum(mx.linalg.norm(b.astype(mx.float32)), 1e-12)
    ).item()


def derivatives(cfg):
    m, _, _ = setup(cfg)
    m.train()
    m.checkpointing = False
    block = m.text.layers[10].linear_attn
    mx.random.seed(103)
    # Set B nonzero so both low-rank factors have meaningful derivatives.
    for _, module in block.named_modules():
        if hasattr(module, "lora_b"):
            module.lora_b = mx.random.normal(module.lora_b.shape) * 0.01
    x = mx.random.normal((2, 32, 1024)).astype(mx.bfloat16)
    cot = mx.random.normal(x.shape)

    def objective(g):
        return (g(g(x)) * cot).mean()

    baseline = None
    native = None
    result = []
    for backend in ["reference", "native", "native", "native"]:
        select(m, backend)
        value, grad = nn.value_and_grad(block, objective)(block)
        grads = dict(tree_flatten(grad))
        mx.eval(value, grad)
        if baseline is None:
            baseline = grads
        result.append(
            dict(
                backend=backend,
                loss=value.item(),
                relative_to_reference={
                    key: relative(v, baseline[key]) for key, v in grads.items()
                },
                relative_to_first_native={
                    key: relative(v, native[key]) for key, v in grads.items()
                }
                if native
                else {},
            )
        )
        if native is None and backend == "native":
            native = grads
    select(m, "reference")
    inner = mx.stop_gradient(block(x))
    input_cot = mx.stop_gradient(mx.grad(lambda h: (block(h) * cot).mean())(inner))
    _, early = nn.value_and_grad(block, lambda g: (g(x) * input_cot).sum())(block)
    _, late = nn.value_and_grad(block, lambda g: (g(inner) * cot).mean())(block)
    late = dict(tree_flatten(late))
    decomposition = {}
    for key, first in tree_flatten(early):
        second = late[key]
        residual = relative(first + second, baseline[key])
        norms = [mx.linalg.norm(v).item() for v in (first, second)]
        if min(norms) <= 0 or residual > 0.035:
            raise AssertionError((key, norms, residual))
        decomposition[key] = dict(site_norms=norms, sum_error=residual)
    select(m, "native")
    docs = rows("validation", "trace")[:2]
    xx, pos, y = batch(docs)
    perturbed = xx
    for i, row in enumerate(docs):
        perturbed = mx.where(
            (mx.arange(xx.shape[1])[None, :] >= row["input_length"])
            & (mx.arange(len(docs))[:, None] == i),
            77,
            perturbed,
        )
    n0 = loss_sum(m, xx, pos, y).item()
    n1 = loss_sum(m, perturbed, pos, y).item()
    if n0 != n1:
        raise AssertionError("Right padding changed scored training positions")
    return dict(
        derivatives=result,
        shared_gradient=decomposition,
        padding_nll_change=(n0 - n1) / y.size,
    )


def profile():
    cfg = config("configs/phase2/p2_a_trace_loop.yaml")
    result = dict(audit=derivatives(cfg), settings=cfg, runs=[])
    docs = rows("validation", "trace")
    for length, b, ck in [
        (128, 1, True),
        (128, 2, True),
        (128, 4, True),
        (128, 4, False),
        (256, 1, True),
        (256, 2, True),
        (512, 1, True),
        (512, 1, False),
    ]:
        if psutil.virtual_memory().available < 1.5 * 2**30:
            raise MemoryError("Free memory before resuming profile")
        m, tok, info = setup(cfg)
        m.train()
        m.checkpointing = ck
        opt = optimizer(cfg)
        # Fill each context with real contiguous repeated V2 token streams for shape timing only.
        contexts = []
        for r in docs[:b]:
            tokens = (r["ids"] * ((length + 1) // len(r["ids"]) + 1))[: length + 1]
            contexts.append(dict(ids=tokens, prefix=length // 2))
        x, p, y = batch(contexts)
        fn = nn.value_and_grad(m, loss_sum)
        mx.reset_peak_memory()
        times = []
        for i in range(4):
            mx.synchronize()
            t = perf_counter()
            value, grad = fn(m, x, p, y)
            grad, _ = optim.clip_grad_norm(tree_map(lambda v: v / y.size, grad), 1.0)
            opt.update(m, grad)
            mx.eval(value, m.trainable_parameters(), opt.state)
            times.append(perf_counter() - t)
        measured = times[1:]
        median = statistics.median(measured)
        result["runs"].append(
            dict(
                length=length,
                batch=b,
                k=2,
                checkpointing=ck,
                seconds=measured,
                median=median,
                mad=statistics.median(abs(s - median) for s in measured),
                input_tps=length * b / median,
                peak=mx.get_peak_memory(),
                rss=psutil.Process().memory_info().rss,
                available=psutil.virtual_memory().available,
                swap=psutil.swap_memory().used,
                backend="native",
                dtype="bf16",
            )
        )
        save(Path("outputs/phase2/profile.json"), result)
        del m, fn, opt, grad, value
        gc.collect()
        mx.clear_cache()
    return result


if __name__ == "__main__":
    profile()
