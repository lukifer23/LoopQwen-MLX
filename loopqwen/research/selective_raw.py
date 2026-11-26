"""Train next-token refinement against two frozen native teachers on heterogeneous raw text."""

import argparse
import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from .. import REVISION
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import load_base
from ..selective_refinement import SelectiveRefinement
from ..tool_model import quantize_prefix
from .cache_language import corpus, measure
from .core import Budget
from .raw_distillation import check_weights, distributions, load_teacher, pad
from .registered import registered


def targets(label, spec, rows, out, budget):
    f = out / (label + "-targets.safetensors")
    meta = out / (label + ".json")
    source = spec["teacher"]["sha256"] if label == "large" else REVISION
    training = digest(Path(spec["training"]))
    if meta.exists():
        r = read(meta)
        if (r["sha256"], r["source"], r["training"]) != (digest(f), source, training):
            raise ValueError("Different frozen teachers or enrolled fitting text")
        return mx.load(str(f))
    net = load_teacher(spec["teacher"]) if label == "large" else base()[0]
    at = perf_counter()
    values = distributions(net, rows, spec["top"], budget, with_gold=True)
    mx.save_safetensors(str(f), values)
    save(
        meta,
        {
            "sha256": digest(f),
            "source": source,
            "training": training,
            "positions": len(rows) * 32,
            "seconds": perf_counter() - at,
        },
    )
    return values


def fit(net, rows, balances, spec, teachers, root, budget):
    at = perf_counter()
    bank = []
    width = max(len(r["ids"]) - 1 for r in rows)
    for i in range(0, len(rows), 8):
        ids = pad(rows[i : i + 8])
        h = net.ordinary(net.prefix_hidden(ids))
        h = mx.pad(h, ((0, 0), (0, width - h.shape[1]), (0, 0)))
        mx.eval(h)
        bank.append(h)
        budget.check()
    bank = mx.concatenate(bank)
    mx.eval(bank)
    prep = perf_counter() - at
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    rng = random.Random(spec["fit_seed"])

    def partial_ce(lp, idx, w):
        chosen = mx.take_along_axis(lp, idx, -1)
        tail = mx.maximum(1 - w.sum(-1), 0)
        other = mx.maximum(1 - mx.exp(chosen).sum(-1), 1e-7)
        return -(w * chosen).sum(-1) - tail * mx.log(other)

    def loss(m, h, ss, gold, large, original):
        refined, _ = m.refine(h)
        select = (mx.arange(h.shape[0])[:, None], ss)
        z = m.project(refined[select]).astype(mx.float32)
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        target = -mx.take_along_axis(lp, gold[..., None], axis=-1)[..., 0]
        blend = mx.sigmoid(
            (large["gold_logp"] - original["gold_logp"]) / spec["temperature"]
        )
        teacher = blend * partial_ce(lp, large["ids"], large["mass"]) + (
            1 - blend
        ) * partial_ce(lp, original["ids"], original["mass"])
        return (
            (1 - spec["gold_weight"]) * teacher + spec["gold_weight"] * target
        ).mean()

    vg = nn.value_and_grad(net, loss)
    log = []
    for step in range(spec["updates"]):
        ix = [
            i
            for group in balances
            for i in rng.sample(group, spec["batch"] // len(balances))
        ]
        rng.shuffle(ix)
        jj = [rng.sample(range(32), spec["per_document"]) for _ in ix]
        ss = [
            [rows[i]["positions"][j] for j in js] for i, js in zip(ix, jj, strict=True)
        ]
        gold = [
            [rows[i]["ids"][p + 1] for p in ps] for i, ps in zip(ix, ss, strict=True)
        ]
        pos = (mx.array(ix)[:, None], mx.array(jj))
        subset = [{k: a[pos] for k, a in t.items()} for t in teachers]
        val, g = vg(net, bank[mx.array(ix)], mx.array(ss), mx.array(gold), *subset)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(net, g)
        mx.eval(val, net.trainable_parameters(), opt.state)
        budget.check()
        if step % 200 == 0 or step + 1 == spec["updates"]:
            log.append([step + 1, val.item()])
            print("selective update", *log[-1], flush=True)
    weights = dict(tree_flatten(net.trainable_parameters()))
    if not all(k.startswith(("gate.", "refinement.self_attn.")) for k in weights):
        raise ValueError("Ordinary decoder was modified by the fit")
    f = root / "refinement.safetensors"
    mx.save_safetensors(str(f), weights)
    save(
        root / "fit.json",
        {
            "weights": digest(f),
            "training": digest(Path(spec["training"])),
            "trainable": sum(v.size for v in weights.values()),
            "parameters": sorted(weights),
            "updates": spec["updates"],
            "seen": spec["updates"] * spec["batch"] * spec["per_document"],
            "prepare_seconds": prep,
            "optimizer_seconds": perf_counter() - at - prep,
            "loss": log,
        },
    )


def material(spec):
    if digest(Path(spec["historical_panel"])) != spec["historical_sha256"]:
        raise ValueError("Changed historical WikiText evaluation")
    old = corpus(dict(spec, material=spec["historical_panel"]))
    fresh = read(spec["material"])["rows"]
    return [dict(row, suite="fresh") for row in fresh] + [
        dict(row, suite="wiki" if row["category"] == "wikitext" else "legacy")
        for row in old
    ]


def outcome(data, items, live):
    metrics = {}
    groups = [
        "pooled",
        "fresh",
        "wiki",
        "legacy",
        *sorted({r["category"] for r in items}),
    ]
    for side, rows in data.items():
        metrics[side] = {}
        for key in groups:
            chosen = [
                (r, w)
                for r, w in zip(rows, items, strict=True)
                if key in ("pooled", w["suite"], w["category"])
            ]
            n = sum(len(r["ids"]) for r, w in chosen)
            metrics[side][key] = {
                "tokens": n,
                "nll": sum(sum(r["nll"]) for r, w in chosen) / n,
                "top1": sum(
                    sum(x == y for x, y in zip(r["ids"], w["ids"][1:], strict=True))
                    for r, w in chosen
                ),
                "seconds": sum(r["seconds"] for r, w in chosen),
                "peak": max(r["peak"] for r, w in chosen),
            }
    a, b = metrics["candidate"], metrics["control"]
    doc_changes = [
        sum(x["nll"]) - sum(y["nll"])
        for x, y, w in zip(data["candidate"], data["control"], items, strict=True)
        if w["suite"] == "fresh"
    ]
    wins = sum(x < 0 for x in doc_changes)
    losses = sum(x > 0 for x in doc_changes)
    p = exact_mcnemar(wins, losses)
    strict = all(
        a[k]["nll"] < b[k]["nll"] and a[k]["top1"] >= b[k]["top1"] for k in groups
    )
    passed = (
        strict
        and a["fresh"]["nll"] <= b["fresh"]["nll"] - 0.015
        and a["wiki"]["nll"] <= b["wiki"]["nll"] - 0.04
        and a["fresh"]["top1"] > b["fresh"]["top1"]
        and p <= 0.05
        and all(live)
        and all(
            a[k]["seconds"] <= b[k]["seconds"] and a[k]["peak"] <= b[k]["peak"]
            for k in ("fresh", "wiki", "legacy")
        )
    )
    return {
        "passed": passed,
        "quality_by_domain": strict,
        "fresh_wins": wins,
        "fresh_losses": losses,
        "p_exact": p,
        "live_parity": live,
        "scores": metrics,
    }


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    rows = read(spec["training"])["rows"]
    items = material(spec)
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        teachers = []
        for label in ("large", "original"):
            teachers.append(targets(label, spec, rows, root, budget))
            gc.collect()
            mx.clear_cache()
        text, tok, _ = load_base()
        quantize_prefix(text, 23, 8)
        mx.random.seed(spec["fit_seed"])
        net = SelectiveRefinement(text, spec["rank"], spec["initial_gate"])
        del text
        net.eval()
        if (root / "fit.json").exists():
            meta = read(root / "fit.json")
            weight = root / "refinement.safetensors"
            if (
                meta["weights"] != digest(weight)
                or meta["training"] != digest(Path(spec["training"]))
                or meta["updates"] != spec["updates"]
            ):
                raise ValueError("Different selective fit")
            check_weights(net, mx.load(str(weight)))
        else:
            groups = [list(range(1024)), list(range(1024, len(rows)))]
            fit(net, rows, groups, spec, teachers, root, budget)
        del teachers
        for side in ("candidate", "control"):
            if side == "control":
                net, tok, _ = base()
            net.eval()
            measure(net, rows[0], False, budget)
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            data[side] = [measure(net, w, False, budget) for w in items]
            packed(root / (side + ".json.gz"), data[side])
            if side == "candidate":
                live = []
                for i in spec["live_indices"]:
                    w = items[i]
                    length = 32
                    c = net.make_cache()
                    ids = w["ids"]
                    mx.eval(net.hidden(mx.array([ids[: length - 1]]), c))
                    h = net.hidden(mx.array([[ids[length - 1]]]), c)
                    live.append(
                        {
                            "id": w["id"],
                            "length": length,
                            "index": i,
                            "prediction": mx.argmax(net.project(h[:, -1:]), -1).item(),
                        }
                    )
                save(root / "live.json", live)
            del net, tok
            gc.collect()
            mx.clear_cache()
        parity = [
            r["prediction"] == data["candidate"][r["index"]]["ids"][r["length"] - 1]
            for r in live
        ]
        save(root / "decision.json", outcome(data, items, parity))
    finally:
        budget.close()


def verify(root, d):
    import math

    spec = read(root / "registration.json")
    items = material(spec)
    rows = read(spec["training"])["rows"]
    meta = read(root / "fit.json")
    assert (
        len(rows) == 2048
        and len(items) == 300
        and meta["training"] == digest(Path(spec["training"]))
    )
    assert (
        meta["weights"] == digest(root / "refinement.safetensors")
        and meta["updates"] == spec["updates"]
        and meta["seen"] == spec["updates"] * spec["batch"] * spec["per_document"]
    )
    for label in ("large", "original"):
        r = read(root / (label + ".json"))
        assert (
            r["sha256"] == digest(root / (label + "-targets.safetensors"))
            and r["positions"] == len(rows) * 32
            and r["training"] == meta["training"]
        )
        assert r["source"] == (
            spec["teacher"]["sha256"] if label == "large" else REVISION
        )
    data = {
        side: unpack(root / (side + ".json.gz")) for side in ("candidate", "control")
    }
    for arm in data.values():
        for w, r in zip(items, arm, strict=True):
            assert (
                w["id"] == r["id"]
                and w["category"] == r["category"]
                and len(r["ids"]) == len(r["nll"]) == len(w["ids"]) - 1
            )
            assert (
                all(math.isfinite(x) and x >= 0 for x in r["nll"])
                and r["seconds"] > 0
                and r["peak"] > 0
            )
    live = read(root / "live.json")
    assert [r["index"] for r in live] == spec["live_indices"] and all(
        items[r["index"]]["id"] == r["id"] and r["length"] == 32 for r in live
    )
    parity = [
        r["prediction"] == data["candidate"][r["index"]]["ids"][r["length"] - 1]
        for r in live
    ]
    assert d == outcome(data, items, parity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
