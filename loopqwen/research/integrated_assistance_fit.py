"""Fit one new shared two-visit expert on real conversations, then combine task displacements."""

import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint
from ..evaluation.mechanism import packed
from ..io import digest, read, save
from ..model_loader import count


def batches(views, seed):
    groups = defaultdict(list)
    database = defaultdict(dict)
    for i, w in enumerate(views):
        if w["group"] == "database":
            # Actions and the original human reply belong to the same real question.
            key = w["id"].rsplit("-", 1)[0]
            database[key][w["kind"]] = i
        else:
            groups[w["group"]].append(i)
    assert all(
        set(d) == {"database_schema", "database_query", "database_answer"}
        for d in database.values()
    )
    groups["database"] = sorted(database)

    def cycle(items, label):
        rng = random.Random(f"{seed}/{label}")
        while True:
            batch = list(items)
            rng.shuffle(batch)
            yield from batch

    streams = {g: cycle(v, g) for g, v in groups.items()}
    while True:
        trio = database[next(streams["database"])]
        yield [
            trio[k] for k in ("database_schema", "database_query", "database_answer")
        ] + [next(streams[g]) for g in ("open", "reasoning", "practice", "rehearsal")]


def positions(labels, visible, cap, seed):
    fixed = {0, len(labels) - 1, visible}
    available = sorted(set(range(len(labels))) - fixed)
    return sorted(
        fixed.union(
            random.Random(seed).sample(available, min(len(available), cap - len(fixed)))
        )
    )


def learning_objective(model, h, p, y):
    sites = model.sites[model.span.start :]
    first = model._execute(h, sites[:1])
    second = model._execute(first, sites[1:])

    def nll(state):
        logits = model.project(model.text.model.norm(state[0, p])).astype(mx.float32)
        return nn.losses.cross_entropy(logits, y, reduction="none")

    reference = mx.stop_gradient(nll(first))
    observed = nll(second)
    gain = reference - observed
    ranking = mx.logaddexp(mx.zeros_like(gain), 0.1 - gain).mean()
    total = observed.mean() + 0.15 * ranking
    return total, observed.mean(), reference.mean(), ranking


def fit(net, tok, s, root, guard, *, prepare):
    start = perf_counter()
    views = read(s["training"])["examples"]
    packed(root / "training-views.json.gz", views)
    bank = prepare(net, tok, {"examples": views}, guard)
    close = tok.convert_tokens_to_ids("</think>")
    visible = []
    for w, (_, _, labels) in zip(views, bank, strict=True):
        at = labels.index(close) + 1
        while at < len(labels) - 1 and not tok.decode([labels[at]]).strip():
            at += 1
        visible.append(at)
    preparation = perf_counter() - start
    optimizer = optim.AdamW(s["lr"], weight_decay=0.01)
    net.checkpointing = True
    net.train()

    grad = nn.value_and_grad(net, learning_objective)
    directory = Path("checkpoints/research") / s["id"]
    directory.mkdir(exist_ok=True, parents=True)
    identity = {
        "training": digest(Path(s["training"])),
        "spec": digest(root / "registration.json"),
    }
    state = {
        "registration": identity,
        "step": 0,
        "rows": [],
        "seconds": 0.0,
        "targets": 0,
    }
    draw = batches(views, s["fit_seed"])
    if (directory / "latest.json").is_file():
        state = checkpoint.restore(directory, net, optimizer)
        assert (
            state["registration"] == identity
            and state["step"] == optimizer.state["step"].item()
        )
        for _ in range(state["step"]):
            next(draw)
    for step in range(state["step"] + 1, s["updates"] + 1):
        guard.check()
        tick = perf_counter()
        batch = next(draw)
        width = len(batch)
        chosen = []
        values = []
        components = []
        total_grad = None
        for i in batch:
            path, offset, y = bank[i]
            h = mx.load(str(path))["hidden"][None]
            pos = positions(
                y, visible[i], s["positions_per_view"], f"{s['fit_seed']}/{step}/{i}"
            )
            loss, g = grad(
                net, h, mx.array(pos) + offset, mx.array([y[j] for j in pos])
            )
            if total_grad is None:
                total_grad = tree_map(lambda t, n=width: t.astype(mx.float32) / n, g)
            else:
                total_grad = tree_map(
                    lambda a, b, n=width: a + b.astype(mx.float32) / n,
                    total_grad,
                    g,
                )
            mx.eval(total_grad, loss)
            observed = [value.item() for value in loss]
            values.append(observed[0])
            components.append(observed[1:])
            chosen.append(pos)
            state["targets"] += len(pos)
        total_grad, norm = optim.clip_grad_norm(total_grad, 1.0)
        mx.eval(total_grad, norm)
        if not all(math.isfinite(v) for v in (*values, norm.item())):
            raise InterruptedError("Nonfinite real-assistance training gradient")
        optimizer.update(net, total_grad)
        mx.eval(net.trainable_parameters(), optimizer.state)
        state["seconds"] += perf_counter() - tick
        state["rows"].append(
            {
                "step": step,
                "views": batch,
                "positions": chosen,
                "losses": values,
                "components": components,
                "gradient_norm": norm.item(),
            }
        )
        if step % 100 == 0:
            state["step"] = step
            checkpoint.write(directory, net, optimizer, state)
            print(
                "Integrated expert update",
                step,
                round(sum(values) / len(values), 4),
                flush=True,
            )
    assert state["step"] == s["updates"]
    weights = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(net.trainable_parameters())))
    latest = checkpoint.location(directory)
    shutil.copyfile(latest / "optimizer.safetensors", root / "optimizer.safetensors")
    packed(root / "optimizer-steps.json.gz", state["rows"])
    save(
        root / "fit.json",
        {
            "weights": digest(weights),
            "passes": 2,
            "trainable": count(net, True),
            "views": digest(root / "training-views.json.gz"),
            "training": identity["training"],
            "updates": state["step"],
            "targets": state["targets"],
            "turns": len(views),
            "prepare_seconds": preparation,
            "optimizer_seconds": state["seconds"],
            "peak": mx.get_peak_memory(),
        },
    )
