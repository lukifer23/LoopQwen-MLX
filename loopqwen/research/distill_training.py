"""Native token-distribution distillation with a separate plain-language retention teacher."""

from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_map

from .. import checkpoint
from ..identity import fingerprint
from ..io import digest
from ..training.exposure import verify_optimizer


def positions(doc, raw, count):
    length = len(doc["ids"]) - 1
    start = 0 if raw else doc["prefix"] - 1
    n = min(count, length - start)
    if n < 1:
        raise ValueError("Empty distillation target")
    return [start + (length - start - 1) * i // max(1, n - 1) for i in range(n)]


def distribution(model, doc, raw, count, k, temperature):
    at = positions(doc, raw, count)
    h = model.hidden(mx.array([doc["ids"][:-1]]))[0, mx.array(at)]
    logits = model.project(h).astype(mx.float32) / temperature
    ids = mx.argpartition(-logits, k - 1, axis=-1)[:, :k]
    logp = nn.log_softmax(logits, axis=-1)
    mass = mx.exp(mx.take_along_axis(logp, ids, axis=-1))
    mx.eval(ids, mass)
    return ids, mass


def bucket_kl(logits, ids, p, temperature):
    logq = nn.log_softmax(logits.astype(mx.float32) / temperature, axis=-1)
    chosen = mx.take_along_axis(logq, ids, axis=-1)
    restp = mx.maximum(1 - mx.sum(p, axis=-1), 0)
    restq = mx.maximum(1 - mx.sum(mx.exp(chosen), axis=-1), 1e-7)
    inside = mx.sum(p * (mx.log(mx.maximum(p, 1e-12)) - chosen), axis=-1)
    outside = restp * (mx.log(mx.maximum(restp, 1e-12)) - mx.log(restq))
    return mx.mean(inside + outside) * temperature**2


def train(model, docs, targets, s, root, budget, source):
    state = {
        "identity": fingerprint([source, s, [r["ids"] for r in docs], digest(targets)]),
        "step": 0,
        "spec": s,
        "parent": source,
        "history": [],
    }
    opt = optim.AdamW(learning_rate=s["lr"], weight_decay=0, bias_correction=True)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "latest.json").exists():
        old = checkpoint.restore(root, model, opt)
        if old["identity"] != state["identity"]:
            raise ValueError("Distillation recovery differs")
        state = old
        verify_optimizer(model, opt, state["step"], s["lr"])
    teacher = mx.load(str(targets))

    def loss(net, x, labels, at, ids, p, prefix, raw):
        h = net.hidden(x)[0]
        kl = bucket_kl(net.project(h[at]), ids, p, s["temperature"])
        if raw:
            return s["retention_weight"] * kl
        hard = mx.fast.cross_entropy(net.project(h[prefix - 1 :]), labels).mean()
        return (1 - s["soft_weight"]) * hard + s["soft_weight"] * kl

    vg = nn.value_and_grad(model, loss)
    model.train()
    model.checkpointing = True
    size = s["group"]
    saved = state["step"]
    try:
        for i in range(saved, s["updates"]):
            start = perf_counter()
            grads = None
            value = 0
            hard_tokens = 0
            positions_seen = 0
            for j in range(i * size, (i + 1) * size):
                budget.check()
                d = docs[j]
                raw = d["raw"]
                at = positions(d, raw, s["positions"])
                v, g = vg(
                    model,
                    mx.array([d["ids"][:-1]]),
                    mx.array(d["ids"][d["prefix"] :]),
                    mx.array(at),
                    teacher[f"{j}.ids"],
                    teacher[f"{j}.mass"],
                    d["prefix"],
                    raw,
                )
                grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
                mx.eval(v, grads)
                value += v.item()
                positions_seen += len(at)
                hard_tokens += 0 if raw else len(d["ids"]) - d["prefix"]
            grads, norm = optim.clip_grad_norm(tree_map(lambda v: v / size, grads), 1)
            if not mx.isfinite(norm).item():
                raise FloatingPointError("Nonfinite distillation gradient")
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            state["history"].append(
                {
                    "loss": value / size,
                    "targets": hard_tokens,
                    "teacher_positions": positions_seen,
                    "seconds": perf_counter() - start,
                    "peak": mx.get_peak_memory(),
                }
            )
            state["step"] = i + 1
            if state["step"] % 16 == 0:
                checkpoint.write(root, model, opt, state)
                saved = state["step"]
    finally:
        if saved != state["step"]:
            checkpoint.write(root, model, opt, state)
    model.eval()
    return state
