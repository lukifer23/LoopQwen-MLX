"""Accumulate the original response-token and paired-margin gradients on Apple MLX."""

import math
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint
from ..io import digest, read, save
from ..kernels.gdn_backend import select
from ..model_loader import count
from .reading_pairs import MatchedExamples


def pack(batch):
    arrays = [mx.load(str(path))["hidden"] for path, _, _ in batch]
    width = max(x.shape[0] for x in arrays)
    h = mx.stack([mx.pad(x, ((0, width - x.shape[0]), (0, 0))) for x in arrays])
    rows, cols, labels = [], [], []
    for i, (_, start, ids) in enumerate(batch):
        rows.extend([i] * len(ids))
        cols.extend(range(start, start + len(ids)))
        labels.extend(ids)
    return h, mx.array(rows), mx.array(cols), mx.array(labels)


def projected(net, h, rows, cols):
    states = net._execute(h, net.sites[net.span.start :])
    return net.project(net.text.model.norm(states[rows, cols])).astype(mx.float32)


def fit(model, tok, spec, out, budget, *, prepare):
    started = perf_counter()
    training = read(spec["training"])
    bank = prepare(model, tok, training, budget)
    paired = MatchedExamples(
        training["examples"], bank, tok, spec["pairs_per_batch"], spec["batch"]
    )
    groups = spec["physical_gradient_groups"]
    if groups != [2, 2, 1, 1] or paired.count != 2 or not spec["individual_margins"]:
        raise ValueError("Unrecognized logical batch design")
    prepared = perf_counter() - started
    select(model, "native")
    model.train()
    model.checkpointing = True
    optimizer = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    rng = random.Random(spec["fit_seed"])
    checkpoints = Path("checkpoints/research") / spec["id"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    binding = digest(out / "registration.json")

    def reference(net, h, r, c, labels, locations):
        logits = projected(net, h, r, c)
        return nn.losses.cross_entropy(logits, labels).mean() + spec[
            "ranking_weight"
        ] * paired.loss(logits, locations, individual=True)

    def partial(net, h, r, c, labels, locations, total):
        logits = projected(net, h, r, c)
        loss = nn.losses.cross_entropy(logits, labels).sum() / total
        if locations is not None:
            loss = (
                loss
                + spec["ranking_weight"]
                * paired.loss(logits, locations, individual=True)
                / paired.count
            )
        return loss

    full_grad = nn.value_and_grad(model, reference)
    stream_grad = nn.value_and_grad(model, partial)
    logs, targets_seen, first = [], 0, 0
    prior_seconds = 0
    parity = None
    if (checkpoints / "latest.json").exists():
        state = checkpoint.restore(checkpoints, model, optimizer)
        if state["registration"] != binding:
            raise ValueError("Optimizer checkpoint registration differs")
        first, logs, parity, prior_seconds = (
            state["step"],
            state["loss"],
            state["parity"],
            state["optimizer_seconds"],
        )
        for _ in range(first):
            targets_seen += sum(len(x[2]) for x in paired.sample(rng))
        if (
            targets_seen != state["targets"]
            or int(optimizer.state["step"].item()) != first
        ):
            raise ValueError(
                "Recovered optimizer differs from registered example exposure"
            )
    started = perf_counter()
    for step in range(first, spec["updates"]):
        budget.check()
        batch = paired.sample(rng)
        total = sum(len(x[2]) for x in batch)
        targets_seen += total
        locations = paired.targets(batch)
        if step == 0:
            reference_loss, ref = full_grad(model, *pack(batch), locations)
            mx.eval(reference_loss, ref)
            mx.clear_cache()
        start = 0
        acc, value = None, 0.0
        for i, size in enumerate(groups):
            subset = batch[start : start + size]
            local = None
            if i < paired.count:
                a, b, x, y = locations[i].tolist()
                offset = sum(len(v[2]) for v in batch[:start])
                local = mx.array([[a - offset, b - offset, x, y]])
            loss, grads = stream_grad(model, *pack(subset), local, total)
            grads = tree_map(lambda g: g.astype(mx.float32), grads)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            mx.eval(loss, acc)
            value += loss.item()
            del grads, loss
            start += size
        if step == 0:
            differences = tree_map(
                lambda a, b: mx.sum(mx.square(a.astype(mx.float32) - b)), ref, acc
            )
            norms = tree_map(lambda a: mx.sum(mx.square(a.astype(mx.float32))), ref)
            norm = lambda tree: math.sqrt(sum(v.item() for _, v in tree_flatten(tree)))
            relative = norm(differences) / max(norm(norms), 1e-30)
            error = abs(reference_loss.item() - value)
            parity = {
                "reference_loss": reference_loss.item(),
                "accumulated_loss": value,
                "absolute_loss_error": error,
                "relative_gradient_l2": relative,
                "targets": total,
            }
            save(out / "gradient-parity.json", parity)
            print("Native first-update equivalence", parity, flush=True)
            del ref, differences, norms
            if not (
                math.isfinite(value)
                and relative <= spec["gradient_relative_l2_max"]
                and error <= spec["loss_absolute_error_max"]
            ):
                raise InterruptedError(
                    "First logical gradient equivalence prerequisite failed"
                )
        if not math.isfinite(value):
            raise InterruptedError("Nonfinite training objective")
        optimizer.update(model, acc)
        mx.eval(model.trainable_parameters(), optimizer.state)
        del acc
        budget.check()
        if (
            step == 0
            or (step + 1) % spec["checkpoint_interval"] == 0
            or step + 1 == spec["updates"]
        ):
            logs.append({"step": step + 1, "loss": value})
            print("fit", logs[-1], flush=True)
            if step:
                checkpoint.write(
                    checkpoints,
                    model,
                    optimizer,
                    {
                        "step": step + 1,
                        "registration": binding,
                        "targets": targets_seen,
                        "parity": parity,
                        "loss": logs,
                        "optimizer_seconds": prior_seconds + perf_counter() - started,
                    },
                )
    elapsed = prior_seconds + perf_counter() - started
    target = out / "tool-adapters.safetensors"
    mx.save_safetensors(str(target), dict(tree_flatten(model.trainable_parameters())))
    save(
        out / "fit.json",
        {
            "passes": model.span.passes,
            "training": digest(Path(spec["training"])),
            "trainable": count(model, True),
            "turns": len(bank),
            "updates": spec["updates"],
            "targets": targets_seen,
            "prepare_seconds": prepared,
            "optimizer_seconds": elapsed,
            "weights": digest(target),
            "loss": logs,
            "parity": parity,
            "physical_gradient_groups": groups,
        },
    )
