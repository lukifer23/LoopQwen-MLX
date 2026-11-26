"""Fit and test natural two-hop document reading with a repeated native Qwen attention layer."""

import argparse
import gc
import random
import re
import string
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..bridge_reader import DocumentReader, features, focus, messages
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_session import prompt
from .core import Budget
from .registered import registered
from .tool_fit import build, fit, restore


def train_reader(model, tok, spec, rows, depth, budget, out):
    mx.random.seed(spec["reader_seed"])
    reader = DocumentReader(model, spec["reader_rank"])
    bank = []
    start = perf_counter()
    for r in rows:
        budget.check()
        x = features(model, tok, r["question"], r["titles"], r["articles"])
        mx.eval(x)
        bank.append(x)
    inputs = mx.stack(bank)
    table = inputs.reshape(-1, inputs.shape[-1]).astype(mx.float32)
    reader.center = mx.mean(table, axis=0)
    reader.scale = mx.rsqrt(mx.var(table, axis=0) + 1e-4)
    mx.eval(reader.center, reader.scale)
    supports = mx.array([r["support"] for r in rows])

    def objective(net, x, s):
        first, last = net.scores(x[:, :-1], x[:, -1], depth)
        return (
            nn.losses.cross_entropy(first, s[:, 0]).mean()
            + nn.losses.cross_entropy(last, s[:, 1]).mean()
        ) / 2

    vg = nn.value_and_grad(reader, objective)
    opt = optim.AdamW(learning_rate=spec["reader_lr"], weight_decay=0.01)
    rng = random.Random(spec["reader_seed"])
    for i in range(spec["reader_updates"]):
        budget.check()
        idx = mx.array(rng.sample(range(len(rows)), spec["reader_batch"]))
        loss, grad = vg(reader, inputs[idx], supports[idx])
        opt.update(reader, grad)
        mx.eval(loss, reader.trainable_parameters(), opt.state)
        if (i + 1) % 200 == 0:
            print("reader", i + 1, loss.item(), flush=True)
    weights = dict(
        tree_flatten(reader.trainable_parameters()),
        center=reader.center,
        scale=reader.scale,
    )
    file = out / "reader.safetensors"
    mx.save_safetensors(str(file), weights)
    save(
        out / "reader.json",
        {
            "weights": digest(file),
            "source": digest(Path(spec["training"])),
            "depth": depth,
            "updates": spec["reader_updates"],
            "trainable": sum(
                v.size for k, v in weights.items() if k not in ("center", "scale")
            ),
            "seconds": perf_counter() - start,
        },
    )
    reader.eval()
    return reader


def reader_from_checkpoint(model, folder, spec):
    meta = read(folder / "reader.json")
    path = folder / "reader.safetensors"
    if (
        meta["weights"] != digest(path)
        or meta["source"] != digest(Path(spec["training"]))
        or meta["updates"] != spec["reader_updates"]
        or meta["depth"] != 2
    ):
        raise ValueError("Different document reader checkpoint")
    reader = DocumentReader(model, spec["reader_rank"])
    arrays = mx.load(str(path))
    expected = dict(
        tree_flatten(reader.trainable_parameters()),
        center=reader.center,
        scale=reader.scale,
    )
    if arrays.keys() != expected.keys() or any(
        x.shape != expected[k].shape for k, x in arrays.items()
    ):
        raise ValueError("Document reader parameter mismatch")
    reader.load_weights(list(arrays.items()), strict=False)
    reader.eval()
    return reader


def units(text):
    value = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", value).split()


def grade(tok, ids, gold):
    complete = ids[-1] in {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
    raw = tok.decode(ids[:-1]) if complete else ""
    pieces = raw.split("</think>")
    visible = pieces[-1].strip() if complete and len(pieces) == 2 else ""
    ans = units(visible)
    ref = units(gold)
    overlap = sum((Counter(ans) & Counter(ref)).values())
    content = bool(ref and (" " + " ".join(ref) + " ") in (" " + " ".join(ans) + " "))
    return {
        "visible": visible,
        "completed": bool(visible),
        "exact": bool(ref and ans == ref),
        "f1": 2 * overlap / (len(ans) + len(ref)) if ans and ref else 0.0,
        "content": content,
    }


def summarize(data, items, spec):
    mets = {}
    for arm, rs in data.items():
        mets[arm] = {
            k: sum(r["score"][k] for r in rs) for k in ("content", "exact", "completed")
        }
        mets[arm].update(
            f1=sum(r["score"]["f1"] for r in rs) / len(rs),
            seconds=sum(r["seconds"] for r in rs),
            tokens=sum(len(r["ids"]) for r in rs),
            peak=max(r["peak"] for r in rs),
        )
    a, b = mets.values()
    wins = sum(
        x["score"]["content"] and not y["score"]["content"]
        for x, y in zip(*data.values(), strict=True)
    )
    losses = sum(
        y["score"]["content"] and not x["score"]["content"]
        for x, y in zip(*data.values(), strict=True)
    )
    pairs = [r["selected"] for r in data["candidate"]]
    both = sum(set(v) == set(w["support"]) for v, w in zip(pairs, items, strict=True))
    p = exact_mcnemar(wins, losses)
    passed = (
        wins - losses >= spec["net_floor"]
        and p <= 0.05
        and a["exact"] >= b["exact"]
        and a["f1"] >= b["f1"] + spec["f1_floor"]
        and both >= spec["retrieval_floor"]
        and a["seconds"] <= b["seconds"]
    )
    return {
        "passed": passed,
        "metrics": mets,
        "gains": wins,
        "losses": losses,
        "p_exact": p,
        "both_documents": both,
        "items": len(items),
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    items = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    data = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    for side, rs in data.items():
        for r, w in zip(rs, items, strict=True):
            text = prompt(tok, messages(w["question"], w["titles"], w["articles"]))
            assert r["id"] == w["id"] and r["prompt_ids"] == tok.encode(
                text, add_special_tokens=False
            )
            assert (
                r["score"] == grade(tok, r["ids"], w["gold"])
                and 0 < len(r["ids"]) <= spec["max_tokens"]
                and r["seconds"] > 0
            )
            assert (
                r["selected"] == []
                if side == "control"
                else len(set(r["selected"])) == 2
                and all(0 <= i < len(w["titles"]) for i in r["selected"])
            )
    assert read(root / "reader.json")["weights"] == digest(root / "reader.safetensors")
    assert read(root / "fit.json")["weights"] == digest(
        root / "tool-adapters.safetensors"
    )
    assert decision == summarize(data, items, spec)


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise ValueError("Existing run")
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(name, spec["limits"])
    items = read(spec["material"])["rows"]
    data = {}
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                local_reg = out / "registration.json"
                if local_reg.exists() and read(local_reg) != spec:
                    raise ValueError("Different local document reading registration")
                save(local_reg, spec)
                if (out / "fit.json").exists():
                    meta = read(out / "fit.json")
                    if (
                        meta["passes"] != 1
                        or meta["updates"] != spec["updates"]
                        or meta["training"] != digest(Path(spec["training"]))
                    ):
                        raise ValueError("Different answer adapter fit")
                    model, tok = restore(out)
                else:
                    model, tok = build(spec, 1)
                    fit(model, tok, spec, out, budget)
                model.eval()
                if (out / "reader.json").exists():
                    reader = reader_from_checkpoint(model, out, spec)
                else:
                    reader = train_reader(
                        model,
                        tok,
                        spec,
                        read(spec["training"])["examples"],
                        2,
                        budget,
                        out,
                    )
            else:
                model, tok, _ = base()
            mx.clear_cache()
            mx.reset_peak_memory()
            rs = []
            for w in items:
                budget.check()
                text = prompt(tok, messages(w["question"], w["titles"], w["articles"]))
                mx.synchronize()
                at = perf_counter()
                view, selected = (
                    focus(
                        model,
                        tok,
                        reader,
                        w["question"],
                        w["titles"],
                        w["articles"],
                        text,
                        2,
                        spec["gain"],
                    )
                    if side == "candidate"
                    else (model, [])
                )
                output = generate(
                    view,
                    tok,
                    text,
                    spec["max_tokens"],
                    stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                    progress=lambda _: budget.check(),
                )
                elapsed = perf_counter() - at
                ids = output["token_ids"]
                rs.append(
                    {
                        "id": w["id"],
                        "prompt_ids": tok.encode(text, add_special_tokens=False),
                        "ids": ids,
                        "selected": selected,
                        "score": grade(tok, ids, w["gold"]),
                        "seconds": elapsed,
                        "peak": mx.get_peak_memory(),
                    }
                )
                del view
                if len(rs) % 8 == 0:
                    print(
                        side,
                        len(rs),
                        sum(r["score"]["content"] for r in rs),
                        flush=True,
                    )
            packed(out / (side + ".json.gz"), rs)
            data[side] = rs
            if side == "candidate":
                del reader
            del model, tok
            gc.collect()
            mx.clear_cache()
        decision = summarize(data, items, spec)
        save(out / "decision.json", decision)
        print(decision, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
