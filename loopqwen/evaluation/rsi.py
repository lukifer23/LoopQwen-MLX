"""Verified two-order trajectories from one real six-pass unroll per presentation."""

import math
from time import perf_counter
import mlx.core as mx
import numpy as np
from ..data import pointer6
from ..data.curriculum import S
from ..training import pointer5 as kernel
from ..training.mechanism import requests
from .mechanism import infer, packed


def score(m, t, arm, rows, folder, weights, guard):
    m.eval()
    by = {q["map_id"]: [] for q in rows}
    start = perf_counter()
    positions = 0
    records = []
    for order in S["orders"]:
        for i in range(0, len(rows), 8):
            guard()
            chunk = [pointer6.ordered(q, order, S["seed"]) for q in rows[i : i + 8]]
            req = requests(chunk, 6, arm, t, {})
            prob, _ = infer(m, arm, [req], 6, False)
            prob = prob.tolist()
            positions += max(len(q["ids"]) for q in req) * len(req)
            for q, pr in zip(chunk, prob, strict=True):
                pred = [int(np.argmax(p)) for p in pr]
                gold = q["states"]
                hit = [x == y for x, y in zip(pred, gold, strict=True)]
                prev = [q["start"], *pred[:-1]]
                prefix = next((j for j, v in enumerate(hit) if not v), 6)
                r = dict(
                    map_id=q["map_id"],
                    order=q["order"],
                    condition=order,
                    map=q["map"],
                    start=q["start"],
                    pred=pred,
                    gold=gold,
                    probabilities=pr,
                    true_p=[p[y] for p, y in zip(pr, gold, strict=True)],
                    entropy=[-sum(x * math.log(max(x, 1e-30)) for x in p) for p in pr],
                    correct=hit,
                    prefix=prefix,
                    first_error=prefix + 1 if prefix < 6 else None,
                    stalled=[
                        a == b and q["map"][b] != b
                        for a, b in zip(pred, prev, strict=True)
                    ],
                )
                by[q["map_id"]].append(r)
                records.append(r)
    hits = np.array([r["correct"] for r in records])
    report = dict(
        seconds=perf_counter() - start,
        maps=len(rows),
        presentations=len(records),
        site_positions=46 * positions,
        accuracy=hits.mean(0).tolist(),
        path=np.cumprod(hits, axis=1).mean(0).tolist(),
        nll=(-np.log(np.maximum(1e-30, [r["true_p"] for r in records])))
        .mean(0)
        .tolist(),
        order_agreement=np.mean(
            [
                [x == y for x, y in zip(a["pred"], b["pred"], strict=True)]
                for a, b in by.values()
            ],
            axis=0,
        ).tolist(),
        peak=mx.get_peak_memory(),
    )
    packed(folder, dict(weights=weights, summary=report, records=records))
    return by, report
