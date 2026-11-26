"""Train a shared passage-relation reader, then freely generate from the original ten documents."""

import argparse
import gc
from pathlib import Path
import random
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..bridge_reader import FocusedDecoder, locate, messages
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..linked_reader import LinkedReader, features
from ..model_loader import local_snapshot
from ..tool_model import quantize_prefix
from ..tool_session import prompt
from .bridge_reading import grade, summarize
from .core import Budget
from .registered import registered
from .tool_fit import restore


def inputs(model, tok, w):
    return features(model, tok, w["question"], w["titles"], w["articles"])


def fit(net, model, tok, spec, root, budget):
    rows = read(spec["training"])["rows"]
    path = root / "reader.safetensors"
    if (root / "fit.json").exists():
        meta = read(root / "fit.json")
        if (
            meta["weights"] != digest(path)
            or meta["training"] != digest(Path(spec["training"]))
            or meta["updates"] != spec["updates"]
        ):
            raise ValueError("Different linked reader fit")
        net.load_weights(str(path), strict=False)
        return
    at = perf_counter()
    values, edges = [], []
    for i, w in enumerate(rows):
        x, e = inputs(model, tok, w)
        values.append(x)
        edges.append(e)
        budget.check()
    x, e = mx.stack(values), mx.stack(edges)
    y = mx.array([w["support"] for w in rows])
    mx.eval(x, e, y)
    del values, edges
    prepared = perf_counter() - at
    rng = random.Random(spec["seed"])
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)

    def loss(m, x, e, y):
        a, b = m(x, e)
        ce = nn.losses.cross_entropy
        return (ce(a, y[:, 0], reduction="mean") + ce(b, y[:, 1], reduction="mean")) / 2

    vg = nn.value_and_grad(net, loss)
    history, norms = [], []
    for i in range(spec["updates"]):
        index = mx.array(rng.sample(range(len(rows)), spec["batch"]))
        value, grad = vg(net, x[index], e[index], y[index])
        if i in (0, spec["updates"] - 1):
            norms.append(
                mx.sqrt(
                    sum(
                        (g.astype(mx.float32) ** 2).sum()
                        for k, g in tree_flatten(grad)
                        if k.startswith("layer.")
                    )
                ).item()
            )
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        opt.update(net, grad)
        mx.eval(value, net.trainable_parameters(), opt.state)
        if i % 200 == 0 or i + 1 == spec["updates"]:
            history.append([i + 1, value.item()])
            print("reader", *history[-1], flush=True)
        budget.check()
    weights = dict(tree_flatten(net.trainable_parameters()))
    mx.save_safetensors(str(path), weights)
    save(
        root / "fit.json",
        dict(
            weights=digest(path),
            training=digest(Path(spec["training"])),
            trainable=sum(v.size for v in weights.values()),
            updates=spec["updates"],
            depth=spec["depth"],
            shared_gradient_norms=norms,
            loss=history,
            prepare_seconds=prepared,
            optimizer_seconds=perf_counter() - at - prepared,
        ),
    )


def pick(first, second):
    left = max(range(len(first)), key=first.__getitem__)
    right = max((i for i in range(len(second)) if i != left), key=second.__getitem__)
    return [left, right]


def outcome(data, items, spec):
    d = summarize(data, items, spec)
    a, b = d["metrics"]["candidate"], d["metrics"]["control"]
    for side, rows in data.items():
        d["metrics"][side]["ttft"] = sum(r["ttft"] for r in rows)
    d["passed"] &= a["peak"] <= b["peak"] and a["ttft"] <= b["ttft"]
    return d


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    parent = Path(spec["answer_checkpoint"])
    if (
        digest(parent / "checksums.json") != spec["answer_archive"]
        or digest(parent / "tool-adapters.safetensors") != spec["answer_weights"]
    ):
        raise ValueError("Different frozen answer adapter")
    budget = Budget(name, spec["limits"])
    items = read(spec["material"])["rows"]
    data = {}
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                model, tok = restore(parent)
                quantize_prefix(model.text, 23, 8)
                mx.random.seed(spec["seed"])
                net = LinkedReader(model, spec["rank"], spec["depth"])
                fit(net, model, tok, spec, root, budget)
                net.eval()
            else:
                model, tok, _ = base()
            mx.eval(model.parameters())
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            rows = []
            for w in items:
                text = prompt(tok, messages(w["question"], w["titles"], w["articles"]))
                mx.synchronize()
                at = perf_counter()
                evidence = {}
                chosen = []
                view = model
                if side == "candidate":
                    x, e = inputs(model, tok, w)
                    a, b = net(x[None], e[None])
                    first, second = a[0].tolist(), b[0].tolist()
                    chosen = pick(first, second)
                    j = chosen[-1]
                    start, end, length = locate(
                        tok, text, w["titles"][j], w["articles"][j]
                    )
                    section = dict(
                        start=start, stop=end, gain=spec["gain"], length=length
                    )
                    evidence = dict(first=first, second=second, section=section)
                    view = FocusedDecoder(model, section)
                selected_at = perf_counter()
                out = generate(
                    view,
                    tok,
                    text,
                    spec["max_tokens"],
                    stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                    progress=lambda _: budget.check(),
                )
                seconds = perf_counter() - at
                ids = out["token_ids"]
                rows.append(
                    dict(
                        id=w["id"],
                        prompt_ids=tok.encode(text, add_special_tokens=False),
                        ids=ids,
                        selected=chosen,
                        evidence=evidence,
                        score=grade(tok, ids, w["gold"]),
                        seconds=seconds,
                        selection_seconds=selected_at - at,
                        decoder_ttft=out["ttft_s"],
                        generation_seconds=out["elapsed_s"],
                        ttft=selected_at - at + out["ttft_s"],
                        peak=mx.get_peak_memory(),
                    )
                )
                del view
                if len(rows) % 8 == 0:
                    print(
                        side,
                        len(rows),
                        sum(r["score"]["content"] for r in rows),
                        flush=True,
                    )
            data[side] = rows
            packed(root / (side + ".json.gz"), rows)
            del model, tok
            if side == "candidate":
                del net
            gc.collect()
            mx.clear_cache()
        save(root / "decision.json", outcome(data, items, spec))
    finally:
        budget.close()


def verify(root, decision):
    spec = read(root / "registration.json")
    meta = read(root / "fit.json")
    assert meta["weights"] == digest(root / "reader.safetensors") and meta[
        "training"
    ] == digest(Path(spec["training"]))
    assert (
        meta["updates"] == spec["updates"]
        and meta["depth"] == spec["depth"]
        and all(n > 0 for n in meta["shared_gradient_norms"])
    )
    items = read(spec["material"])["rows"]
    training = read(spec["training"])["rows"]
    assert len(training) == 1536 and len(items) == 48

    def identities(rows):
        return {
            s.casefold().strip()
            for w in rows
            for s in (w["question"], *(w["titles"][i] for i in w["support"]))
        }

    assert identities(training).isdisjoint(identities(items))
    tok = load_tokenizer(local_snapshot())
    data = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    for side, rows in data.items():
        for w, r in zip(items, rows, strict=True):
            text = prompt(tok, messages(w["question"], w["titles"], w["articles"]))
            assert w["id"] == r["id"] and r["prompt_ids"] == tok.encode(
                text, add_special_tokens=False
            )
            assert (
                r["score"] == grade(tok, r["ids"], w["gold"])
                and 0 < len(r["ids"]) <= spec["max_tokens"]
                and 0 < r["ttft"] <= r["seconds"]
            )
            assert (
                r["selection_seconds"] >= 0
                and r["generation_seconds"] >= r["decoder_ttft"]
            )
            assert r["ttft"] == r["selection_seconds"] + r["decoder_ttft"]
            if side == "candidate":
                e = r["evidence"]
                assert r["selected"] == pick(e["first"], e["second"])
                i = r["selected"][-1]
                a, b, n = locate(tok, text, w["titles"][i], w["articles"][i])
                assert e["section"] == dict(
                    start=a, stop=b, length=n, gain=spec["gain"]
                )
            else:
                assert r["selected"] == [] and r["evidence"] == {}
    assert outcome(data, items, spec) == decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
