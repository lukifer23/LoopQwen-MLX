"""Evaluate the full-vocabulary preference gradient in bounded position tiles on MLX."""

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from .. import checkpoint
from ..io import digest, read, save
from .table_reentry_fit import accumulate


def start(net, opt, root, s):
    parent = Path(s["resume_archive"])
    inventory = read(parent / "checksums.json")
    for name in (
        "partial-state.json",
        "partial-adapters.safetensors",
        "partial-optimizer.safetensors",
    ):
        assert digest(parent / name) == inventory[name]
    where = root / "initial-state"
    where.mkdir(exist_ok=True)
    for src, dst in (
        ("partial-state.json", "state.json"),
        ("partial-adapters.safetensors", "adapters.safetensors"),
        ("partial-optimizer.safetensors", "optimizer.safetensors"),
    ):
        shutil.copyfile(parent / src, where / dst)
    state = checkpoint.restore(where, net.readout, opt)
    assert state["registration"] == digest(parent / "registration.json") and state[
        "step"
    ] == s["resume_step"] == len(state["log"])
    save(
        root / "continuation.json",
        {
            "parent": str(parent),
            "archive": digest(parent / "checksums.json"),
            "weights": inventory["partial-adapters.safetensors"],
            "optimizer": inventory["partial-optimizer.safetensors"],
            "history": inventory["partial-state.json"],
            "step": state["step"],
        },
    )
    for p in where.iterdir():
        p.unlink()
    where.rmdir()
    state["registration"] = digest(root / "registration.json")
    return state


def evaluator(net, s):
    def nll(readout, features, labels):
        logp = nn.log_softmax(
            net.project(net.text.model.norm(readout(features))).astype(mx.float32)
        )
        return -mx.take_along_axis(logp, labels[:, None], axis=-1)[:, 0], logp

    def tile_loss(readout, features, labels, pos, coeff):
        loss, logp = nll(readout, features, labels)
        ordinary = features[:, 0 if s["passes"] == 2 else -1]
        logt = mx.stop_gradient(
            nn.log_softmax(
                net.project(net.text.model.norm(ordinary)).astype(mx.float32)
            )
        )
        kl = (mx.sum(mx.exp(logt) * (logt - logp), axis=-1) * pos).sum()
        ce = (loss * pos).sum()
        return (loss * coeff).sum() + s["teacher_kl"] * kl, ce, kl

    grad = nn.value_and_grad(net.readout, tile_loss)
    width = s["position_tile"]

    def micro(readout, features, labels, positive, pairs):
        pairloss = mx.array(0.0)
        coeff = positive
        if pairs.shape[0]:
            # Recompute only the changing token positions to obtain the derivative
            # of each whole-pair softplus before the tiled backward pass.
            selected = np.flatnonzero(np.asarray(pairs).any(axis=0)).astype(np.int32)
            p = pairs[:, mx.array(selected)]
            pieces = []
            for i in range(0, len(selected), width):
                ids = mx.array(selected[i : i + width])
                vals = nll(readout, features[ids], labels[ids])[0]
                mx.eval(vals)
                pieces.append(vals)
            margin = s["margin"] - p @ mx.concatenate(pieces)
            pairloss = nn.softplus(margin).sum()
            coeff = mx.stop_gradient(
                positive - s["preference_weight"] * (mx.sigmoid(margin) @ pairs)
            )
            mx.eval(coeff, pairloss)
        total = None
        ce = mx.array(0.0)
        kl = mx.array(0.0)
        for i in range(0, labels.size, width):
            end = i + width
            (_, c, k), g = grad(
                readout, features[i:end], labels[i:end], positive[i:end], coeff[i:end]
            )
            total = accumulate(total, g, 1)
            ce, kl = ce + c, kl + k
            mx.eval(total, ce, kl)
        return (
            ce + s["teacher_kl"] * kl + s["preference_weight"] * pairloss,
            ce,
            kl,
            pairloss,
        ), total

    return micro


def preflight(net, s, root, index):
    from .conditional_fit import load_batch
    from .contrastive_fit import objective

    native = nn.value_and_grad(net.readout, objective(net, s))
    streamed = evaluator(net, s)
    records = []
    for length, needs_pair in ((26, False), (46, True), (75, True)):
        i = next(
            i
            for i, r in enumerate(index)
            if bool(r["negative"]) == needs_pair
            and length
            <= r["positions"]
            + (len(r["negative"][0]["negative_positions"]) if needs_pair else 0)
            < length + 10
        )
        inputs, _ = load_batch(index, Path("."), [(i, 0 if needs_pair else None)])
        a, g = native(net.readout, *inputs)
        mx.eval(a, g)
        b, h = streamed(net.readout, *inputs)
        mx.eval(b, h)
        aa = mx.concatenate([v.flatten() for _, v in tree_flatten(g)])
        bb = mx.concatenate([v.flatten() for _, v in tree_flatten(h)])
        relative = mx.linalg.norm(aa - bb) / mx.linalg.norm(aa)
        cosine = (aa @ bb) / (mx.linalg.norm(aa) * mx.linalg.norm(bb))
        error = max(
            abs(x.item() - y.item()) / max(1.0, abs(x.item()))
            for x, y in zip(a, b, strict=True)
        )
        records.append(
            {
                "id": index[i]["id"],
                "positions": inputs[1].size,
                "preferences": inputs[-1].shape[0],
                "objective_relative": error,
                "gradient_relative": relative.item(),
                "gradient_cosine": cosine.item(),
            }
        )
    save(root / "gradient-parity.json", records)
    if not all(
        r["objective_relative"] <= s["objective_tolerance"]
        and r["gradient_relative"] <= s["gradient_tolerance"]
        and r["gradient_cosine"] >= s["gradient_cosine"]
        for r in records
    ):
        raise InterruptedError(
            "Full-vocabulary tiled derivatives missed registered native agreement"
        )
