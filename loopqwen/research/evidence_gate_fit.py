"""Fit the request-level choice on labeled fitting dialogues and reserve whole questions."""

import random
from time import perf_counter

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from ..io import digest, save
from ..shared_session import EvidenceGate, evidence_features


def fit_gate(decoder, tok, rows, spec, root, budget):
    at = perf_counter()
    embedding = decoder.text.model.embed_tokens
    xs = []
    for w in rows:
        h = evidence_features(embedding, tok, w["question"], w["schema"])
        mx.eval(h)
        xs.append(h)
        budget.check()
    xs = mx.stack(xs)
    labels = mx.array([w["external"] for w in rows], mx.float32)
    mx.eval(xs, labels)
    mx.random.seed(spec["fit_seed"] + 1)
    gate = EvidenceGate(decoder.args.hidden_size, spec["gate_width"])
    optimizer = optimizers.AdamW(learning_rate=0.00025, weight_decay=0.01)

    def loss(net, x, y):
        return nn.losses.binary_cross_entropy(
            net(x), y, with_logits=True, reduction="mean"
        )

    vg = nn.value_and_grad(gate, loss)
    rng = random.Random(spec["fit_seed"] + 1)
    train = [
        [i for i, w in enumerate(rows) if w["split"] == "train" and w["external"] == z]
        for z in (False, True)
    ]
    for _ in range(spec["gate_updates"]):
        ix = mx.array([i for group in train for i in rng.sample(group, 32)])
        value, grad = vg(gate, xs[ix], labels[ix])
        optimizer.update(gate, grad)
        mx.eval(gate.trainable_parameters(), optimizer.state, value)
        budget.check()
    held = [i for i, w in enumerate(rows) if w["split"] == "validation"]
    scores = mx.sigmoid(gate(xs[mx.array(held)])).tolist()
    validation = [
        {"id": rows[i]["id"], "external": rows[i]["external"], "p": p}
        for i, p in zip(held, scores, strict=True)
    ]
    path = root / "router.safetensors"
    params = dict(tree_flatten(gate.trainable_parameters()))
    mx.save_safetensors(str(path), params)
    save(
        root / "router.json",
        {
            "weights": digest(path),
            "trainable": sum(v.size for v in params.values()),
            "train": [len(g) for g in train],
            "updates": spec["gate_updates"],
            "validation": validation,
            "seconds": perf_counter() - at,
        },
    )
