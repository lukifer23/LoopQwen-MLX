"""Cohort-bound categorical evaluation. Record probabilities before independently scoring."""

import json
import math
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import checkpoint
from ..data.pointer5 import ROOT, rows, cycle
from ..identity import fingerprint
from ..io import read, save, digest
from ..training.pointer5 import (
    OUT,
    SPEC,
    REC,
    architecture,
    identity,
    request_rows,
    inputs,
    outputs,
    root_for,
    sites,
)


def summarize(records):
    metrics = []
    for j in range(len(records[0]["pred"])):
        correct = complete = consistent = over = 0
        bins = [[0, 0, 0.0] for _ in range(10)]
        nll = 0.0
        wrap = [0, 0]
        fresh = [0, 0]
        for r in records:
            p = r["probabilities"][j]
            pred = r["pred"][j]
            gold = r["gold"][j]
            hit = pred == gold
            correct += hit
            complete += r["pred"][: j + 1] == r["gold"][: j + 1]
            prev = r["pred"][j - 1] if j else r["start"]
            consistent += pred == r["map"][prev]
            if j:
                over += pred == r["gold"][j - 1]
            conf = max(p)
            b = bins[min(9, int(conf * 10))]
            b[0] += 1
            b[1] += hit
            b[2] += conf
            nll -= math.log(max(1e-38, p[gold]))
            group = wrap if j + 1 >= r["cycle"] else fresh
            group[0] += hit
            group[1] += 1
        size = len(records)
        metrics.append(
            dict(
                pass_number=j + 1,
                correct=correct,
                n=size,
                accuracy=correct / size,
                trajectory=complete,
                consistent=consistent,
                preceding_terminal=over if j else None,
                nll=nll / size,
                ece=sum(abs(b[1] - b[2]) for b in bins) / size,
                calibration=bins,
                wrapped=wrap,
                nonwrapping=fresh,
            )
        )
    return metrics


def evaluate(m, tok, arm, items, depths=(1, 2, 3)):
    m.eval()
    cache = {}
    times = {}
    rec = [
        dict(
            map=r["map"],
            map_id=r["map_id"],
            example_id=r["example_id"],
            start=r["start"],
            order=r["order"],
            cycle=cycle(r),
            gold=r["states"][: max(depths)],
            pred=[],
            probabilities=[],
        )
        for r in items
    ]
    for n in depths:
        begin = perf_counter()
        positions = 0
        for i in range(0, len(items), 8):
            grouped = request_rows(items[i : i + 8], n, arm, tok, cache)
            x, last = inputs(grouped)
            z = outputs(m, x, last, n, arm)[-1]
            for r, p in zip(
                rec[i : i + 8],
                mx.softmax(z.astype(mx.float32), axis=-1).tolist(),
                strict=True,
            ):
                r["probabilities"].append(p)
                r["pred"].append(max(range(8), key=p.__getitem__))
            positions += x.size
        times[str(n)] = dict(
            seconds=perf_counter() - begin,
            site_positions=positions * sites(arm, n),
            maps=len(items),
        )
    return dict(records=rec, metrics=summarize(rec), runtime=times)


def milestone(seed, arm, step, m=None, tok=None):
    root = root_for(seed, arm)
    gen = root / read(root / f"at-{step}.json")["directory"]
    st = read(gen / "state.json")
    weight = digest(gen / "adapters.safetensors")
    if st["step"] != step or st["identity"] != identity(st["config"], arm):
        raise ValueError("Milestone provenance mismatch")
    items = rows("development")[: 384 if step >= 480 else 96]
    eid = fingerprint(
        dict(
            training=st["identity"],
            weights=weight,
            data=fingerprint(items),
            eval=digest(Path(__file__)),
            step=step,
        )
    )
    path = OUT / f"{seed}/{arm}-{step}.json"
    if path.exists():
        result = read(path)
        if result["identity"] != eid:
            raise ValueError(
                "Incompatible saved evaluation; use a new result namespace"
            )
        return result
    if m is None:
        m, tok, _, _ = architecture(seed, arm, read(OUT / "profile.json")["selected"])
        checkpoint.restore(gen, m)
    result = evaluate(m, tok, arm, items)
    result.update(
        identity=eid,
        weights=weight,
        optimizer=digest(gen / "optimizer.safetensors"),
        training=st["identity"],
        step=step,
        arm=arm,
        capacity=st["capacity"],
    )
    save(path, result)
    print(
        arm,
        step,
        "dev",
        [v["correct"] for v in result["metrics"]],
        "/",
        len(items),
        flush=True,
    )
    return result


def installed(seed, arm="recurrent"):
    result = milestone(seed, arm, 480)
    priors = read(ROOT / "manifest.json")["prior"]
    return all(
        m["accuracy"]
        >= max(p["conditional"], p["prior"], p["majority"]) + SPEC["prior_margin"]
        for m, p in zip(result["metrics"], priors, strict=True)
    ) and result["metrics"][2]["correct"] > sum(
        r["pred"][0] == r["gold"][2] for r in result["records"]
    )
