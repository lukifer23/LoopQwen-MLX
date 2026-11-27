"""Fit two shared visits to current evidence while protecting actual older-task responses."""

import math
import random
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
from .file_evidence_fit import pack, projected
from .linked_rollin_fit import loss_fn
from .native_function_data import token_pair


def semantic(tok, w):
    _, y = token_pair(tok, w)
    close = y.index(tok.convert_tokens_to_ids("</think>"))
    places = list(range(close + 1, len(y)))
    bad = None
    if w["alternative"] is not None:
        _, alternative = token_pair(tok, {**w, "target": w["alternative"]})
        bad = next(
            ((j, b) for j, (a, b) in enumerate(zip(y, alternative)) if a != b), None
        )
        assert bad is None or bad[0] > close
    reasoning = sum(bool(tok.decode([t]).strip()) for t in y[:close])
    return places, bad, reasoning


def batches(views, seed):
    rng = random.Random(seed)
    kinds = {}
    for i, w in enumerate(views):
        kinds.setdefault(w["kind"], []).append(i)
    while True:
        ref = rng.sample(kinds["protected"], 2)
        human = "human_first" if rng.random() < 0.35 else "human_followup"
        yield [rng.choice(kinds[k]) for k in (human, "action", "interpret")], ref


def reference_loss(net, h, r, c, y):
    return nn.losses.cross_entropy(projected(net, h, r, c), y).mean()


def inner(a, b):
    vals = tree_flatten(
        tree_map(lambda x, y: mx.sum(x.astype(mx.float32) * y.astype(mx.float32)), a, b)
    )
    return mx.stack([v for _, v in vals]).sum()


def protected_update(net, adam, main_grad, ref_grad, ref_args, before, tolerance):
    original = net.trainable_parameters()
    proposed = adam.apply_gradients(main_grad, original)
    delta = tree_map(
        lambda a, b: a.astype(mx.float32) - b.astype(mx.float32), proposed, original
    )
    dot = inner(delta, ref_grad)
    g2 = inner(ref_grad, ref_grad)
    penalty = mx.maximum(dot, 0) / mx.maximum(g2, 1e-30)
    direction = tree_map(
        lambda d, g: d - penalty * g.astype(mx.float32), delta, ref_grad
    )
    proposed_l2 = mx.sqrt(inner(delta, delta))
    mx.eval(direction, dot, g2, proposed_l2)
    if not all(math.isfinite(t.item()) for t in (dot, g2, proposed_l2)):
        raise InterruptedError("Nonfinite native update direction")
    for k in range(5):
        factor = 0.5**k
        candidate = tree_map(
            lambda p, d, step_size=factor: (
                p.astype(mx.float32) + step_size * d
            ).astype(p.dtype),
            original,
            direction,
        )
        net.update(candidate)
        after = reference_loss(net, *ref_args).item()
        if not math.isfinite(after):
            net.update(original)
            raise InterruptedError("Nonfinite protected objective on a proposed step")
        if after <= before + tolerance:
            realized = tree_map(
                lambda a, b: a.astype(mx.float32) - b.astype(mx.float32),
                candidate,
                original,
            )
            signed = inner(realized, ref_grad)
            step_l2 = mx.sqrt(inner(realized, realized))
            mx.eval(signed, step_l2)
            if not all(math.isfinite(t.item()) for t in (signed, step_l2)):
                net.update(original)
                raise InterruptedError("Nonfinite applied native update")
            return {
                "projected": dot.item() > 0,
                "raw_directional": dot.item(),
                "actual_directional": signed.item(),
                "proposed_step_l2": proposed_l2.item(),
                "applied_step_l2": step_l2.item(),
                "factor": factor,
                "before": before,
                "after": after,
            }
    net.update(original)
    return {
        "projected": dot.item() > 0,
        "raw_directional": dot.item(),
        "actual_directional": 0.0,
        "proposed_step_l2": proposed_l2.item(),
        "applied_step_l2": 0.0,
        "factor": 0.0,
        "before": before,
        "after": before,
    }


def fit(net, tok, s, root, guard, *, prepare):
    began = perf_counter()
    views = read(s["training"])["examples"]
    packed(root / "training-views.json.gz", views)
    bank = prepare(net, tok, {"examples": views}, guard)
    meta = [semantic(tok, w) for w in views]
    prep = perf_counter() - began
    main_gradient = nn.value_and_grad(net, loss_fn)
    ref_gradient = nn.value_and_grad(net, reference_loss)
    net.train()
    net.checkpointing = True
    adam = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    bind = {
        "registration": digest(root / "registration.json"),
        "views": digest(root / "training-views.json.gz"),
    }
    folder = Path("checkpoints/research") / s["id"]
    folder.mkdir(exist_ok=True, parents=True)
    state = dict(
        **bind,
        step=0,
        steps=[],
        optimizer_seconds=0.0,
        targets=0,
        explanation_targets=0,
        protected_targets=0,
    )
    draws = batches(views, s["fit_seed"])
    if (folder / "latest.json").exists():
        state = checkpoint.restore(folder, net, adam)
        assert (
            all(state[k] == v for k, v in bind.items())
            and adam.state["step"].item() == state["step"]
        )
        for _ in range(state["step"]):
            next(draws)
    for step in range(state["step"] + 1, s["updates"] + 1):
        guard.check()
        began = perf_counter()
        active, protected = next(draws)
        ref_args = pack([bank[i] for i in protected])
        before, rg = ref_gradient(net, *ref_args)
        mx.eval(before, rg)
        before = before.item()
        if not math.isfinite(before):
            raise InterruptedError("Nonfinite protected objective")
        grad = None
        loss = 0.0
        margins = []
        for i, weight in zip(active, (0.4, 0.27, 0.33), strict=True):
            places, bad, n = meta[i]
            (local, gaps), g = main_gradient(
                net,
                *pack([bank[i]]),
                mx.array(places),
                *(bad if bad else (-1, 0)),
                return_margins=True,
            )
            grad = (
                tree_map(lambda z, scale=weight: z.astype(mx.float32) * scale, g)
                if grad is None
                else tree_map(
                    lambda a, b, scale=weight: a + scale * b.astype(mx.float32), grad, g
                )
            )
            mx.eval(local, gaps, grad)
            if bad:
                j, foil = bad
                first, second = gaps.tolist()
                if not math.isfinite(first) or not math.isfinite(second):
                    raise InterruptedError("Nonfinite training token margin")
                margins.append(
                    {
                        "view": i,
                        "position": j,
                        "target": bank[i][2][j],
                        "foil": foil,
                        "first": first,
                        "second": second,
                    }
                )
            loss += weight * local.item()
            state["targets"] += len(bank[i][2])
            state["explanation_targets"] += n
        if not math.isfinite(loss):
            raise InterruptedError("Nonfinite new-task objective")
        # Rehearsal supplies a descent direction when the new-task gradient is already tangential.
        grad = tree_map(lambda a, b: 0.85 * a + 0.15 * b.astype(mx.float32), grad, rg)
        measure = protected_update(
            net, adam, grad, rg, ref_args, before, s["reference_tolerance"]
        )
        mx.eval(net.trainable_parameters(), adam.state)
        state["optimizer_seconds"] += perf_counter() - began
        state["protected_targets"] += len(ref_args[-1])
        state["steps"].append(
            dict(
                step=step,
                active=active,
                protected=protected,
                main_loss=loss,
                margins=margins,
                **measure,
            )
        )
        if step == 1 or step % 100 == 0:
            print(
                "Protected update", step, round(loss, 6), measure["factor"], flush=True
            )
        if step % 100 == 0:
            state["step"] = step
            checkpoint.write(folder, net, adam, state)
            save(root / "optimizer-directions.json", state["steps"])
            if (
                sum(m["factor"] > 0 for m in state["steps"][-100:])
                < s["min_applied_per_100"]
            ):
                raise InterruptedError(
                    "Insufficient applied native updates in this training block"
                )
    dest = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(dest), dict(tree_flatten(net.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "weights": digest(dest),
            "training": digest(Path(s["training"])),
            "views": bind["views"],
            "passes": 2,
            "trainable": count(net, True),
            "updates": state["step"],
            "turns": len(views),
            "targets": state["targets"],
            "explanation_targets": state["explanation_targets"],
            "protected_targets": state["protected_targets"],
            "prepare_seconds": prep,
            "optimizer_seconds": state["optimizer_seconds"],
            "peak": mx.get_peak_memory(),
            "projected": sum(x["projected"] for x in state["steps"]),
            "applied": sum(x["factor"] > 0 for x in state["steps"]),
        },
    )
