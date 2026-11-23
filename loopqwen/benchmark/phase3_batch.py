"""Matched-cohort real optimizer-step timing, including optional exact microbatch splitting."""

from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import psutil
from mlx.utils import tree_map

from .. import checkpoint
from ..config import config
from ..data.sampling import prepare
from ..data.tokenize import batch
from ..identity import fingerprint
from ..io import save
from ..training.losses import loss_sum
from ..training.optimizers import optimizer


def profile(policy, cap=None):
    c = config("outputs/phase3/configs/p3_direct.yaml")
    # Fix the complete cohort before reordering it. No case can draw a cheaper example.
    pool = prepare({**c, "bucket": False})[0][:80]
    key = (
        (lambda r: r["bucket_length"])
        if policy == "matched_instance_order"
        else (lambda r: (r["protocol"], r["input_length"]))
    )
    ordered = sorted(pool, key=key)
    model, _, _ = checkpoint.setup(c)
    opt = optimizer(c)
    grad = nn.value_and_grad(model, loss_sum)
    model.train()
    mx.set_cache_limit(384 * 1024**2)

    def update(items):
        start = perf_counter()
        group = []
        batches = []
        for r in items:
            longest = max([r["input_length"], 16] + [v["input_length"] for v in group])
            if cap and (len(group) + 1) * longest > cap and group:
                batches.append(group)
                group = []
            group.append(r)
        batches.append(group)
        grads = None
        losses = None
        labels = active = padded = 0
        shapes = []
        for g in batches:
            x, p, y = batch(g)
            v, d = grad(model, x, p, y)
            grads = d if grads is None else tree_map(lambda a, b: a + b, grads, d)
            losses = v if losses is None else losses + v
            labels += y.size
            padded += x.size
            active += sum(r["input_length"] for r in g)
            shapes.append(list(x.shape))
            mx.eval(grads, losses)
        grads = tree_map(lambda w: w / labels, grads)
        grads, norm = optim.clip_grad_norm(grads, c["clip"])
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, norm)
        return dict(
            seconds=perf_counter() - start,
            inputs=active,
            positions=padded,
            targets=labels,
            loss=losses.item() / labels,
            norm=norm.item(),
            shapes=shapes,
            available=psutil.virtual_memory().available,
        )

    for i in (0, 4):
        update(pool[i : i + 4])
    mx.reset_peak_memory()
    history = [update(ordered[i : i + 4]) for i in range(0, len(pool), 4)]
    secs = sum(r["seconds"] for r in history)
    inputs = sum(r["inputs"] for r in history)
    padded = sum(r["positions"] for r in history)
    record = dict(
        cohort=fingerprint(sorted((r["id"], r["protocol"]) for r in ordered)),
        config=c,
        policy=policy,
        cap=cap,
        history=history,
        warmup_updates=2,
        update_seconds=secs,
        median_seconds=median(r["seconds"] for r in history),
        targets_s=sum(r["targets"] for r in history) / secs,
        useful_inputs_s=inputs / secs,
        padding=1 - inputs / padded,
        peak=mx.get_peak_memory(),
        rss=psutil.Process().memory_info().rss,
        swap=psutil.swap_memory().used,
    )
    save(Path("outputs/phase3") / f"batch-{policy}-{cap}.json", record)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", choices=["matched_instance_order", "protocol_and_length"])
    p.add_argument("--cap", type=int)
    a = p.parse_args()
    profile(a.policy, a.cap)
