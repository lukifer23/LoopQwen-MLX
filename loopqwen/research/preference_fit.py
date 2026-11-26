"""Update the shared native action decoder against a fixed preference reference."""

import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from ..io import digest, read, save
from ..model_loader import count
from .tool_fit import features


def batch_inputs(examples):
    width = max(h.shape[0] for h, _, _ in examples)
    padded = mx.stack(
        [mx.pad(h, ((0, width - h.shape[0]), (0, 0))) for h, _, _ in examples]
    )
    rows, positions, labels = [], [], []
    for i, (_, offset, ids) in enumerate(examples):
        rows.extend([i] * len(ids))
        positions.extend(range(offset, offset + len(ids)))
        labels.extend(ids)
    return padded, mx.array(rows), mx.array(positions), mx.array(labels)


def log_probs(model, hidden, rows, positions, labels):
    states = model._execute(hidden, model.sites[model.span.start :])
    logits = model.project(model.text.model.norm(states[rows, positions])).astype(
        mx.float32
    )
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    return mx.zeros((hidden.shape[0],), mx.float32).at[rows].add(-losses)


def train(model, tok, spec, root, budget):
    model.eval()
    training = read(spec["training"])
    prefs = training["preferences"]
    t0 = perf_counter()
    rows = [
        {"messages": w["messages"], "visible": w[k], "thinking": ""}
        for w in prefs
        for k in ("chosen", "rejected")
    ]
    bank = features(model, tok, {"examples": rows}, budget)
    replay = features(model, tok, {"examples": training["replay"]}, budget)
    prepared = perf_counter() - t0
    t0 = perf_counter()
    reference = []
    for j in range(0, len(bank), 2):
        budget.check()
        prob = log_probs(model, *batch_inputs(bank[j : j + 2]))
        reference.append(prob.tolist())
    ref_seconds = perf_counter() - t0
    save(
        root / "reference.json", {"scores": reference, "parent": spec["initial_sha256"]}
    )
    train_ix = [i for i, w in enumerate(prefs) if w["split"] == "train"]
    val_ix = [i for i, w in enumerate(prefs) if w["split"] == "validation"]
    batch = spec["batch"]
    beta = spec["beta"]

    def loss(net, h, r, p, y, reference_margin, lengths):
        sums = log_probs(net, h, r, p, y)
        chosen = sums[: 2 * batch : 2]
        rejected = sums[1 : 2 * batch : 2]
        delta = chosen - rejected - reference_margin
        dpo = mx.logaddexp(0.0, -beta * delta).mean()
        chosen_ce = -(chosen / lengths[: 2 * batch : 2]).mean()
        replay_ce = -(sums[-1] / lengths[-1])
        return (
            dpo + spec["chosen_weight"] * chosen_ce + spec["replay_weight"] * replay_ce
        )

    optimizer = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    vg = nn.value_and_grad(model, loss)
    rng = random.Random(spec["fit_seed"])
    used = 0
    sample_counts = {i: 0 for i in train_ix}
    logs = []
    t0 = perf_counter()
    for step in range(spec["updates"]):
        budget.check()
        ix = rng.sample(train_ix, batch)
        for i in ix:
            sample_counts[i] += 1
        examples = [bank[2 * i + j] for i in ix for j in (0, 1)] + [rng.choice(replay)]
        lengths = [len(ids) for _, _, ids in examples]
        used += sum(lengths)
        margins = mx.array([reference[i][0] - reference[i][1] for i in ix])
        v, g = vg(model, *batch_inputs(examples), margins, mx.array(lengths))
        optimizer.update(model, g)
        mx.eval(v, model.trainable_parameters(), optimizer.state)
        if step % 80 == 0 or step + 1 == spec["updates"]:
            logs.append([step + 1, v.item()])
            print("preference update", logs[-1], flush=True)
    seconds = perf_counter() - t0
    receipts = []
    for i in val_ix:
        budget.check()
        pair = log_probs(model, *batch_inputs(bank[2 * i : 2 * i + 2])).tolist()
        receipts.append(
            {
                "index": i,
                "source_line": prefs[i]["source_line"],
                "group": prefs[i]["group"],
                "parent": reference[i],
                "fitted": pair,
            }
        )
    save(root / "validation.json", receipts)
    weights = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(model.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "passes": model.span.passes,
            "trainable": count(model, True),
            "weights": digest(weights),
            "training": digest(Path(spec["training"])),
            "updates": spec["updates"],
            "targets": used,
            "train_pairs": len(train_ix),
            "sampled_pairs": len([i for i in sample_counts if sample_counts[i]]),
            "validation_pairs": len(val_ix),
            "replay_turns": len(replay),
            "prepare_seconds": prepared,
            "reference_seconds": ref_seconds,
            "optimizer_seconds": seconds,
            "loss": logs,
        },
    )
    del bank, replay, optimizer, vg
    gc.collect()
    mx.clear_cache()
