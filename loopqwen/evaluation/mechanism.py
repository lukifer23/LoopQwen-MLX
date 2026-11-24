"""Cohort- and token-bound multi-order categorical evaluation with shared recurrent exits."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from .. import checkpoint
from ..data import pointer5, pointer6
from ..identity import fingerprint
from ..io import digest, read
from ..training import pointer5 as kernel
from ..training.mechanism import generation, heads, model_for, provenance, requests

SPEC = pointer6.SPEC


def packed(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(
        gzip.compress(
            json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), mtime=0
        )
    )
    tmp.replace(path)


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def load(arm, step):
    gen = generation(arm, step)
    m, tok, c = model_for(arm)
    st = checkpoint.restore(gen, m)
    if st["step"] != step or fingerprint(st["provenance"]) != fingerprint(
        provenance(c, arm)
    ):
        raise ValueError("Checkpoint implementation or configuration changed")
    m.eval()
    return m, tok, st, digest(gen / "adapters.safetensors")


def trace_stats(previous, h, last):
    batch = mx.arange(h.shape[0])
    a = previous.astype(mx.float32)
    b = h.astype(mx.float32)
    v = b[batch, last]
    u = a[batch, last]
    rms = lambda x: mx.mean(x.square(), axis=-1).sqrt()
    return mx.stack(
        [
            rms(v),
            rms(v - u),
            mx.sum(u * v, axis=-1)
            / (mx.linalg.norm(u, axis=-1) * mx.linalg.norm(v, axis=-1) + 1e-12),
            rms(b[:, :-1]).mean(axis=-1),
            rms(b[:, :-1] - a[:, :-1]).mean(axis=-1),
        ],
        axis=-1,
    )


def infer(m, arm, batches, depth, observe):
    scores = []
    stats = []
    if arm.kind == "recurrent":
        x, last = kernel.inputs(batches[0])
        h = m.prefix_hidden(x)
        for _ in range(depth):
            prev = h
            h = m.shared_block(h)
            scores.append(kernel.readout(m, h, last))
            if observe:
                stats.append(trace_stats(prev, h, last))
    elif arm.kind == "heads":
        x, last = kernel.inputs(batches[0])
        scores = heads(m, next(m.block_states(x, 1)), last)[:depth]
    else:
        for j, group in enumerate(batches, 1):
            x, last = kernel.inputs(group)
            scores.append(kernel.outputs(m, x, last, j, arm.kind)[-1])
    p = mx.softmax(mx.stack(scores, axis=1).astype(mx.float32), axis=-1)
    measures = mx.stack(stats, axis=1) if stats else None
    return p, measures


def evaluate(arm, step, items, conditions, depth, out, *, batch_size=16, observe=False):
    m, tok, st, weights = load(arm, step)
    allrows = [
        (condition, pointer6.ordered(r, condition, SPEC["evaluation_seed"]))
        for condition in conditions
        for r in items
    ]
    nreq = depth if arm.kind in ("middle", "full") else 1
    cache = {}
    prepared = [
        requests([r for _, r in allrows], n, arm, tok, cache)
        for n in range(1, nreq + 1)
    ]
    streams = [fingerprint([r["ids"] for r in group]) for group in prepared]
    identity = dict(
        weights=weights,
        training=st["identity"],
        assets=st["provenance"]["assets"],
        runtime=digest(Path(__file__)),
        dependencies=st["provenance"]["base"]["dependencies"],
        config=st["config"],
        registration=fingerprint(SPEC),
        items=fingerprint(allrows),
        token_streams=streams,
        depth=depth,
        conditions=conditions,
        batch=batch_size,
        observe=observe,
        readout=arm.kind,
    )
    if out.is_file():
        prev = unpack(out)
        if fingerprint(prev["identity"]) != fingerprint(identity):
            raise ValueError(f"Incompatible evaluation at {out}")
        return prev
    mx.reset_peak_memory()
    mx.synchronize()
    begin = perf_counter()
    records = []
    for i in range(0, len(allrows), batch_size):
        batches = [group[i : i + batch_size] for group in prepared]
        p, act = infer(m, arm, batches, depth, observe)
        probs = p.tolist()
        activation = act.tolist() if act is not None else [None] * len(probs)
        for (condition, r), pr, values in zip(
            allrows[i : i + batch_size], probs, activation, strict=True
        ):
            record = {
                k: r[k] for k in ("map", "map_id", "example_id", "start", "order")
            }
            record.update(
                condition=condition, gold=r["states"][:depth], probabilities=pr
            )
            if values is not None:
                record["activations"] = values
            records.append(record)
    result = dict(
        identity=identity,
        records=records,
        seconds=perf_counter() - begin,
        peak=mx.get_peak_memory(),
        trained_step=step,
    )
    packed(out, result)
    print(
        arm.name,
        step,
        conditions[0],
        len(records),
        "requests",
        round(result["seconds"], 1),
        "s",
        flush=True,
    )
    return result


def evaluate_arm(arm):
    items = pointer5.rows("development")
    for step in arm.milestones:
        final = step == arm.milestones[-1]
        evaluate(
            arm,
            step,
            items if final else items[:96],
            SPEC["orders"] if final else ["canonical", "random0"],
            3,
            arm.output / f"orders-{step}.json.gz",
        )
    if arm.kind == "recurrent":
        evaluate(
            arm,
            arm.milestones[-1],
            items,
            ["canonical", "random0"],
            6,
            arm.output / "deep.json.gz",
            observe=True,
        )
