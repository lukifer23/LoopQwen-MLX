"""Train a causal neural-cache interpolation and compare raw language quality and cost."""

import argparse
import gc
import json
import math
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim

from ..data import retention3
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import read, save
from ..model_loader import load_base
from ..neural_cache import RecurrentMemory, weights
from .core import Budget
from .registered import registered


def corpus(spec):
    held = read(spec["material"])["rows"]
    meta = retention3.build()
    if meta["sha256"] != spec["retention_sha256"]:
        raise ValueError("Changed ordinary language retention data")
    old = [
        json.loads(line)
        for line in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    if not all(r["prefix"] == 1 for r in old):
        raise ValueError("Unexpected masked targets")
    return held + old


def mask(start, stop, count):
    x = mx.arange(start, stop)[:, None]
    y = mx.arange(count)[None, :]
    return (y < x) & (y >= x - 256)


def fit(net, spec, budget):
    bank = []
    start = perf_counter()
    for row in read(spec["training"])["rows"]:
        budget.check()
        ids = mx.array(row["ids"])
        h, z = net.representations(ids[None, :-1])
        # Native probability and similarities are frozen. Train only the three mixing coefficients.
        p = []
        target = ids[1:]
        for i in range(0, len(h), 32):
            probs = mx.softmax(
                net.decoder.project(h[i : i + 32]).astype(mx.float32), axis=-1
            )
            p.append(mx.take_along_axis(probs, target[i : i + 32, None], axis=-1)[:, 0])
        record = (mx.concatenate(p), z @ z[:-1].T, target[:, None] == target[None, :-1])
        mx.eval(record)
        bank.append(record)
    rng = random.Random(spec["seed"])
    params = net.parameters
    opt = optim.AdamW(learning_rate=0.025, weight_decay=0.0)

    def loss(p, native, similar, match):
        allowed = mask(0, native.shape[1], similar.shape[-1])
        dist, gate = weights(p, similar, allowed)
        mixed = (1 - gate) * native + gate * mx.sum(dist * match, axis=-1)
        return -mx.log(mx.maximum(mixed, 1e-35)).mean()

    grad = mx.value_and_grad(loss)
    for i in range(spec["updates"]):
        batch = rng.sample(bank, 4)
        cost, g = grad(params, *[mx.stack(v) for v in zip(*batch, strict=True)])
        params = opt.apply_gradients({"p": g}, {"p": params})["p"]
        mx.eval(params, opt.state, cost)
        budget.check()
    net.parameters = params
    return {
        "parameters": params.tolist(),
        "seconds": perf_counter() - start,
        "train_loss": cost.item(),
    }


def measure(net, row, candidate, budget):
    ids = mx.array(row["ids"])
    target = ids[1:]
    mx.synchronize()
    at = perf_counter()
    if candidate:
        h, z = net.representations(ids[None, :-1])
        decoder = net.decoder
    else:
        decoder = net
        h = net.hidden(ids[None, :-1])[0]
    predictions = []
    losses = []
    for start in range(0, len(h), 32):
        stop = min(start + 32, len(h))
        probs = mx.softmax(decoder.project(h[start:stop]).astype(mx.float32), axis=-1)
        if candidate:
            probs = net.correct(
                probs, z[start:stop], z[:-1], target[:-1], mask(start, stop, len(z) - 1)
            )
        gold = mx.take_along_axis(probs, target[start:stop, None], axis=-1)[:, 0]
        predictions.extend(mx.argmax(probs, axis=-1).tolist())
        losses.extend((-mx.log(mx.maximum(gold, 1e-35))).tolist())
    elapsed = perf_counter() - at
    budget.check()
    return {
        "id": row["id"],
        "category": row["category"],
        "ids": predictions,
        "nll": losses,
        "seconds": elapsed,
        "peak": mx.get_peak_memory(),
    }


def decide(data, items, spec, parity):
    scores = {}
    for side, rows in data.items():
        scores[side] = {}
        for cat in ["pooled", "retention", *sorted({w["category"] for w in items})]:
            subset = [
                (r, w)
                for r, w in zip(rows, items, strict=True)
                if cat == "pooled"
                or r["category"] == cat
                or (cat == "retention" and r["category"] != "wikitext")
            ]
            n = sum(len(r["nll"]) for r, w in subset)
            scores[side][cat] = {
                "tokens": n,
                "nll": sum(sum(r["nll"]) for r, w in subset) / n,
                "top1": sum(
                    sum(a == b for a, b in zip(r["ids"], w["ids"][1:], strict=True))
                    for r, w in subset
                ),
                "seconds": sum(r["seconds"] for r, w in subset),
                "peak": max(r["peak"] for r, w in subset),
            }
    a, b = scores["candidate"], scores["control"]
    wins = losses = 0
    for x, y in zip(*data.values(), strict=True):
        if x["category"] == "wikitext":
            wins += sum(x["nll"]) < sum(y["nll"])
            losses += sum(y["nll"]) < sum(x["nll"])
    p = exact_mcnemar(wins, losses)
    passed = (
        all(
            a[k]["nll"] <= b[k]["nll"] - spec["nll_margin"]
            for k in ("wikitext", "retention")
        )
        and all(a[k]["top1"] >= b[k]["top1"] for k in ("wikitext", "retention"))
        and all(
            a[k]["nll"] <= b[k]["nll"] + 0.02
            for k in a
            if k not in ("wikitext", "retention", "pooled")
        )
        and a["pooled"]["top1"] > b["pooled"]["top1"]
        and p <= 0.05
        and wins > losses
        and a["pooled"]["peak"] <= b["pooled"]["peak"]
        and a["pooled"]["seconds"] <= b["pooled"]["seconds"]
        and all(parity)
    )
    return {
        "passed": passed,
        "scores": scores,
        "article_wins": wins,
        "article_losses": losses,
        "p_exact": p,
        "live_parity": parity,
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    items = corpus(spec)
    data = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    for rows in data.values():
        for r, w in zip(rows, items, strict=True):
            assert r["id"] == w["id"] and r["category"] == w["category"]
            assert len(r["ids"]) == len(r["nll"]) == len(w["ids"]) - 1 and all(
                math.isfinite(n) and n >= 0 for n in r["nll"]
            )
            assert r["seconds"] > 0 and r["peak"] > 0
    live = read(root / "live-prefix.json")
    parity = [
        s["prediction"] == data["candidate"][i]["ids"][s["length"] - 1]
        for i, s in enumerate(live)
    ]
    assert len(live) == 6 and all(
        s["id"] == items[i]["id"] and s["length"] == spec["prefix_length"]
        for i, s in enumerate(live)
    )
    assert decision == decide(data, items, spec, parity)


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise ValueError("Existing run")
    out.mkdir(exist_ok=True, parents=True)
    budget = Budget(name, spec["limits"])
    data = {}
    items = corpus(spec)
    try:
        for side in ("candidate", "control"):
            candidate = side == "candidate"
            if candidate:
                text, _, _ = load_base()
                net = RecurrentMemory(text, [math.log(12), -3.0, 1.0])
                del text
                meta = fit(net, spec, budget)
                save(out / "fit.json", meta)
            else:
                net = base()[0]
            # Compile the scoring path on a disjoint training excerpt before taking timings.
            measure(net, read(spec["training"])["rows"][0], candidate, budget)
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            data[side] = [measure(net, r, candidate, budget) for r in items]
            packed(out / (side + ".json.gz"), data[side])
            if candidate:
                live = [
                    {
                        "id": r["id"],
                        "length": spec["prefix_length"],
                        "prediction": net.generate(
                            r["ids"][: spec["prefix_length"]], 1
                        )[0],
                    }
                    for r in items[:6]
                ]
                save(out / "live-prefix.json", live)
            del net
            gc.collect()
            mx.clear_cache()
        parity = [
            s["prediction"] == data["candidate"][i]["ids"][s["length"] - 1]
            for i, s in enumerate(live)
        ]
        decision = decide(data, items, spec, parity)
        save(out / "decision.json", decision)
        print(decision, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
