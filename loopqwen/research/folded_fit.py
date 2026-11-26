"""Distill a removed depth on real trajectories using only native hidden-state targets."""

import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from ..evaluation.mechanism import packed
from ..io import digest, read, save


def corpus(decoder, removed, rows, budget):
    banks = [[], [], []]
    start = decoder.span.start
    for i in range(0, len(rows), 4):
        batch = rows[i : i + 4]
        n = max(len(w["ids"]) for w in batch)
        ids = mx.array([w["ids"] + [0] * (n - len(w["ids"])) for w in batch])
        x = decoder.prefix_hidden(ids)
        first = decoder._execute(x, [(start, 0)])
        raw = decoder._execute(first, [(start, 1)])
        target = removed(first, mask=None, cache=None)
        index = (
            mx.arange(len(batch))[:, None],
            mx.array([w["positions"] for w in batch]),
        )
        states = [a[index] for a in (first, raw, target)]
        mx.eval(states)
        for bank, v in zip(banks, states, strict=True):
            bank.append(v)
        budget.check()
        if (i + 4) % 256 == 0:
            print("folded teacher documents", i + 4, flush=True)
    result = [mx.concatenate(bank) for bank in banks]
    mx.eval(result)
    return result


def errors(predicted, target, x):
    scale = mx.mean(mx.square(x.astype(mx.float32)), axis=-1) + 1e-6
    return (
        mx.mean(
            mx.square(predicted.astype(mx.float32) - target.astype(mx.float32)), axis=-1
        )
        / scale
    )


def fit(assistant, repair, removed, spec, out, budget):
    started = perf_counter()
    enrolled = read(spec["training"])["rows"]
    training = [w for w in enrolled if w["split"] == "train"]
    validation = [w for w in enrolled if w["split"] == "validation"]
    train = corpus(assistant.language, removed, training, budget)
    valid = corpus(assistant.language, removed, validation, budget)
    prep = perf_counter() - started
    train = [a.reshape(-1, a.shape[-1]) for a in train]
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)

    def loss(module, x, raw, target):
        return errors(module(x, raw), target, x).mean()

    vg = nn.value_and_grad(repair, loss)
    rng = random.Random(spec["seed"])
    logs = []
    started = perf_counter()
    for step in range(spec["updates"]):
        ix = mx.array(rng.sample(range(train[0].shape[0]), spec["batch"]))
        value, grads = vg(repair, *[v[ix] for v in train])
        opt.update(repair, grads)
        mx.eval(repair.parameters(), opt.state, value)
        budget.check()
        if step % 300 == 0 or step + 1 == spec["updates"]:
            logs.append([step + 1, value.item()])
            print("folded update", *logs[-1], flush=True)
    elapsed = perf_counter() - started
    x, raw, target = valid
    adjusted = repair(x, raw)
    lookup = {
        name: errors(y, target, x).mean(axis=1).tolist()
        for name, y in (("uncorrected", raw), ("skip", x), ("learned", adjusted))
    }
    packed(
        out / "hidden-validation.json.gz",
        [
            {
                "id": w["id"],
                "domain": w["domain"],
                **{k: v[i] for k, v in lookup.items()},
            }
            for i, w in enumerate(validation)
        ],
    )
    weights = dict(tree_flatten(repair.parameters()))
    path = out / "folded-repair.safetensors"
    mx.save_safetensors(str(path), weights)
    save(
        out / "fit.json",
        {
            "training": digest(Path(spec["training"])),
            "weights": digest(path),
            "updates": spec["updates"],
            "parameters": sum(a.size for a in weights.values()),
            "train_documents": len(training),
            "validation_documents": len(validation),
            "observed_training_positions": train[0].shape[0],
            "sampled_positions": spec["batch"] * spec["updates"],
            "prepare_seconds": prep,
            "optimizer_seconds": elapsed,
            "loss": logs,
        },
    )
