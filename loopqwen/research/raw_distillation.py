"""Distill raw next-token probabilities from a larger frozen Base into recurrent shared depth."""

import argparse
import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download
from mlx_lm import load

from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..model_loader import load_base, local_snapshot
from ..recurrence import Span
from ..tool_model import quantize_prefix
from ..training.adapters import attach
from .cache_language import corpus, decide, measure, verify as verify_scores
from .core import Budget
from .registered import registered


def pad(rows):
    seq = [r["ids"][:-1] for r in rows]
    n = max(map(len, seq))
    return mx.array([s + [0] * (n - len(s)) for s in seq])


def teacher_targets(spec, rows, out, budget):
    t = spec["teacher"]
    d = Path(
        snapshot_download(t["repo"], revision=t["revision"], local_files_only=True)
    )
    path = d / t["shard"]
    if digest(path) != t["sha256"] or digest(d / "tokenizer.json") != digest(
        local_snapshot() / "tokenizer.json"
    ):
        raise ValueError("Teacher or raw token vocabulary differs")
    outer, _ = load(str(d))
    net = LoopedModel(outer.language_model, Span(passes=1))
    net.eval()
    del outer
    indices = []
    masses = []
    start = perf_counter()
    for i in range(0, len(rows), 4):
        batch = rows[i : i + 4]
        h = net.hidden(pad(batch))
        pos = mx.array([r["positions"] for r in batch])
        states = h[mx.arange(len(batch))[:, None], pos]
        z = net.project(states).astype(mx.float32)
        winners = mx.argpartition(-z, spec["top"] - 1, axis=-1)[..., : spec["top"]]
        mass = mx.exp(
            mx.take_along_axis(z, winners, axis=-1)
            - mx.logsumexp(z, axis=-1, keepdims=True)
        )
        mx.eval(winners, mass)
        indices.append(winners)
        masses.append(mass)
        budget.check()
        if i % 128 == 0:
            print("teacher documents", i, flush=True)
    idx = mx.concatenate(indices)
    p = mx.concatenate(masses)
    mx.eval(idx, p)
    f = out / "teacher-targets.safetensors"
    mx.save_safetensors(str(f), {"ids": idx, "mass": p})
    save(
        out / "teacher.json",
        dict(
            sha256=digest(f),
            teacher=t["sha256"],
            training=digest(Path(spec["training"])),
            positions=len(rows) * 32,
            seconds=perf_counter() - start,
        ),
    )
    return idx, p


def student(spec):
    text, tok, _ = load_base()
    quantize_prefix(text, 23, 8)
    mx.random.seed(spec["fit_seed"])
    net = LoopedModel(
        text,
        Span(start=23, length=1, passes=spec["passes"], mode="stabilized"),
        alpha=spec["alpha"],
    )
    attach(net, rank=spec["rank"], alpha=2 * spec["rank"])
    net.freeze(keys=["alpha_logit"], recurse=False)
    net.eval()
    mx.eval(net.parameters())
    return net, tok


def fit(spec, net, rows, targets, out, budget):
    at = perf_counter()
    bank = []
    n = max(len(r["ids"]) - 1 for r in rows)
    for i in range(0, len(rows), 8):
        x = pad(rows[i : i + 8])
        h = net.prefix_hidden(x)
        h = mx.pad(h, ((0, 0), (0, n - h.shape[1]), (0, 0)))
        mx.eval(h)
        bank.append(h)
        budget.check()
    bank = mx.concatenate(bank)
    mx.eval(bank)
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    rng = random.Random(spec["fit_seed"])
    tid, mass = targets
    seen = 0

    def objective(model, h, ss, targets, ti, w):
        y = model._execute(h, model.sites[23:])
        h = model.text.model.norm(y[mx.arange(h.shape[0])[:, None], ss])
        z = model.project(h).astype(mx.float32)
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(lp, ti, axis=-1)
        tail = mx.maximum(1 - w.sum(-1), 0)
        left = mx.maximum(1 - mx.exp(chosen).sum(-1), 1e-7)
        teacher = -(w * chosen).sum(-1) - tail * mx.log(left)
        gold = -mx.take_along_axis(lp, targets[..., None], axis=-1)[..., 0]
        return ((1 - spec["gold_weight"]) * teacher + spec["gold_weight"] * gold).mean()

    vg = nn.value_and_grad(net, objective)
    logs = []
    for step in range(spec["updates"]):
        ix = rng.sample(range(len(rows)), spec["batch"])
        jj = [rng.sample(range(32), spec["per_document"]) for _ in ix]
        ss = [
            [rows[i]["positions"][j] for j in js] for i, js in zip(ix, jj, strict=True)
        ]
        gold = [
            [rows[i]["ids"][p + 1] for p in ps] for i, ps in zip(ix, ss, strict=True)
        ]
        idx = mx.array(ix)[:, None]
        j = mx.array(jj)
        loss, g = vg(
            net,
            bank[mx.array(ix)],
            mx.array(ss),
            mx.array(gold),
            tid[idx, j],
            mass[idx, j],
        )
        opt.update(net, g)
        mx.eval(loss, net.trainable_parameters(), opt.state)
        budget.check()
        seen += len(ix) * len(jj[0])
        if step % 100 == 0 or step + 1 == spec["updates"]:
            logs.append([step + 1, loss.item()])
            print("student update", *logs[-1], flush=True)
    f = out / "raw-adapter.safetensors"
    weights = dict(tree_flatten(net.trainable_parameters()))
    mx.save_safetensors(str(f), weights)
    save(
        out / "fit.json",
        dict(
            training=digest(Path(spec["training"])),
            weights=digest(f),
            seen=seen,
            updates=spec["updates"],
            trainable=sum(v.size for v in weights.values()),
            seconds=perf_counter() - at,
            loss=logs,
        ),
    )
    net.eval()


def check_weights(net, tensors):
    expected = dict(tree_flatten(net.trainable_parameters()))
    if tensors.keys() != expected.keys() or any(
        v.shape != expected[k].shape for k, v in tensors.items()
    ):
        raise ValueError("Raw adapter parameter mismatch")
    net.load_weights(list(tensors.items()), strict=False)
    net.eval()


def verify(root, decision):
    spec = read(root / "registration.json")
    verify_scores(root, decision)
    t = read(root / "teacher.json")
    f = read(root / "fit.json")
    assert (
        f["training"] == t["training"] == digest(Path(spec["training"]))
        and t["teacher"] == spec["teacher"]["sha256"]
    )
    assert t["sha256"] == digest(root / "teacher-targets.safetensors") and f[
        "weights"
    ] == digest(root / "raw-adapter.safetensors")
    assert (
        f["seen"] == spec["updates"] * spec["batch"] * spec["per_document"]
        and t["positions"] == len(read(spec["training"])["rows"]) * 32
    )


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    out.mkdir(exist_ok=True, parents=True)
    rows = read(spec["training"])["rows"]
    items = corpus(spec)
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        if (out / "teacher.json").exists():
            rec = read(out / "teacher.json")
            f = out / "teacher-targets.safetensors"
            if (
                rec["training"] != digest(Path(spec["training"]))
                or rec["teacher"] != spec["teacher"]["sha256"]
                or digest(f) != rec["sha256"]
            ):
                raise ValueError("Different previously collected teacher")
            saved = mx.load(str(f))
            targets = saved["ids"], saved["mass"]
            del saved
        else:
            targets = teacher_targets(spec, rows, out, budget)
        gc.collect()
        mx.clear_cache()
        net, tok = student(spec)
        if (out / "fit.json").exists():
            f = out / "raw-adapter.safetensors"
            rec = read(out / "fit.json")
            if (
                rec["training"] != digest(Path(spec["training"]))
                or rec["weights"] != digest(f)
                or rec["updates"] != spec["updates"]
            ):
                raise ValueError("Different saved fit")
            check_weights(net, mx.load(str(f)))
        else:
            fit(spec, net, rows, targets, out, budget)
        del targets
        gc.collect()
        mx.clear_cache()
        live = []
        for side in ("candidate", "control"):
            if side == "control":
                net, tok, _ = base()
            net.eval()
            measure(net, rows[0], False, budget)
            mx.clear_cache()
            mx.reset_peak_memory()
            data[side] = [measure(net, r, False, budget) for r in items]
            packed(out / (side + ".json.gz"), data[side])
            if side == "candidate":
                for r in items[:6]:
                    l = spec["prefix_length"]
                    c = net.make_cache()
                    ids = r["ids"][:l]
                    mx.eval(net.hidden(mx.array([ids[:-1]]), c))
                    h = net.hidden(mx.array([[ids[-1]]]), c)
                    pred = mx.argmax(net.project(h[:, -1:]), -1).item()
                    live.append(dict(id=r["id"], length=l, prediction=pred))
                    del c
                save(out / "live-prefix.json", live)
            del net, tok
            gc.collect()
            mx.clear_cache()
        parity = [
            r["prediction"] == data["candidate"][i]["ids"][r["length"] - 1]
            for i, r in enumerate(live)
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
