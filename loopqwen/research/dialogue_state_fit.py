"""Teach the first native pass to represent which required fields have been supplied."""

import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from ..io import digest, read, save
from ..tool_session import compact, prompt
from .preference_fit import batch_inputs
from .tool_fit import features


def fields(tok, row):
    known = row.get("known", {})
    if not known:
        return [], []
    text = prompt(tok, row["messages"])
    fragment = '"required":' + compact(list(known))
    start = text.index(fragment) + len('"required":[')
    encoded = tok._tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    assert encoded["input_ids"] == tok.encode(text, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    positions = []
    for key in known:
        quoted = compact(key)
        start = text.index(quoted, start)
        end = start + len(quoted) - 1
        positions.append(next(i for i, (a, b) in enumerate(offsets) if a < end <= b))
        start = end + 1
    return positions, list(known.values())


class StateFit(nn.Module):
    def __init__(self, decoder, width):
        super().__init__()
        self.decoder = decoder
        size = decoder.args.hidden_size
        self.query = nn.Linear(size, width, bias=True)
        self.key = nn.Linear(size, width, bias=True)
        self.bias = mx.array(0.0)
        self.width = width

    def read_fields(self, first, query, slots):
        normalized = self.decoder.text.model.norm(first)
        i = mx.arange(first.shape[0])
        q = self.query(normalized[i, query].astype(mx.float32))
        keys = self.key(normalized[i[:, None], slots].astype(mx.float32))
        return (keys * q[:, None]).sum(-1) * self.width**-0.5 + self.bias


def prepare(state, tok, rows, budget, locator=fields):
    bank = features(state.decoder, tok, {"examples": rows}, budget)
    return [(entry, *locator(tok, r)) for entry, r in zip(bank, rows, strict=True)]


def tensors(samples):
    entries, positions, targets = zip(*samples, strict=True)
    width = max(1, max(map(len, positions)))
    return (
        *batch_inputs(entries),
        mx.array([e[1] for e in entries]),
        mx.array([x + [0] * (width - len(x)) for x in positions]),
        mx.array([x + [False] * (width - len(x)) for x in targets], mx.float32),
        mx.array([[True] * len(x) + [False] * (width - len(x)) for x in positions]),
    )


def train(
    decoder, tok, spec, root, budget, *, locator=fields, kinds=("ask", "call", "replay")
):
    mx.random.seed(spec["fit_seed"])
    decoder.eval()
    state = StateFit(decoder, spec["state_width"])
    data = read(spec["training"])
    rows = data["examples"]
    start = perf_counter()
    bank = prepare(state, tok, rows, budget, locator)
    held = prepare(state, tok, data["validation"], budget, locator)
    prepared = perf_counter() - start
    layer = decoder.span.start
    assert (
        decoder.span.length == 1
        and decoder.span.passes == 2
        and decoder.span.mode == "stabilized"
    )

    def loss(m, h, r, p, y, q, slots, known, valid):
        first = m.decoder._execute(h, [(layer, 0)])
        estimated = m.read_fields(first, q, slots)
        bce = nn.losses.binary_cross_entropy(
            estimated, known, with_logits=True, reduction="none"
        )
        aux = (bce * valid).sum() / mx.maximum(valid.sum(), 1)
        second = m.decoder._execute(first, [(layer, 1)])
        word = m.decoder.project(m.decoder.text.model.norm(second[r, p])).astype(
            mx.float32
        )
        return nn.losses.cross_entropy(word, y).mean() + spec["field_weight"] * aux

    optimizer = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    vg = nn.value_and_grad(state, loss)
    rng = random.Random(spec["fit_seed"])
    strata = [
        [i for i, r in enumerate(rows) if r.get("kind", "replay") == k] for k in kinds
    ]
    seen = set()
    targets = 0
    t0 = perf_counter()
    logs = []
    for n in range(spec["updates"]):
        # Balance question/execute targets, using authentic older responses as rehearsal.
        ix = [rng.choice(group) for group in strata] + [rng.choice(strata[n % 2])]
        seen.update(ix)
        pack = tensors([bank[i] for i in ix])
        cost, grad = vg(state, *pack)
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        optimizer.update(state, grad)
        mx.eval(cost, optimizer.state, state.trainable_parameters())
        targets += pack[3].size
        budget.check()
        if n % 100 == 0 or n + 1 == spec["updates"]:
            logs.append([n + 1, cost.item()])
            print("state update", logs[-1], flush=True)
    seconds = perf_counter() - t0
    validation = []
    for w, sample in zip(data["validation"], held, strict=True):
        h, _, _, _, q, slots, _, _ = tensors([sample])
        first = decoder._execute(h, [(layer, 0)])
        probs = mx.sigmoid(state.read_fields(first, q, slots)).tolist()[0]
        validation.append(
            {
                "id": w["id"],
                "known": w["known"],
                "probabilities": dict(zip(w["known"], probs, strict=True)),
            }
        )
    save(root / "validation.json", validation)
    adapter = root / "tool-adapters.safetensors"
    head = root / "state-head.safetensors"
    params = dict(tree_flatten(decoder.trainable_parameters()))
    mx.save_safetensors(str(adapter), params)
    extra = {
        k: v
        for k, v in tree_flatten(state.trainable_parameters())
        if not k.startswith("decoder.")
    }
    mx.save_safetensors(str(head), extra)
    save(
        root / "fit.json",
        {
            "training": digest(Path(spec["training"])),
            "weights": digest(adapter),
            "head": digest(head),
            "trainable": sum(p.size for p in params.values()),
            "state_head_parameters": sum(p.size for p in extra.values()),
            "turns": len(rows),
            "validation_turns": len(held),
            "seen_examples": len(seen),
            "targets": targets,
            "passes": 2,
            "updates": spec["updates"],
            "prepare_seconds": prepared,
            "optimizer_seconds": seconds,
            "loss": logs,
        },
    )
    del bank, held, optimizer, vg, state
    gc.collect()
    mx.clear_cache()
