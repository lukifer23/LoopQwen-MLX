"""Fit the recurrent next-token model and evaluate previously unused human passages."""

import argparse
import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import load_base, local_snapshot
from ..passage_copy import CopyReader, PassageRequest, request
from ..tool_model import quantize_prefix, tool_view
from .bridge_reading import grade
from .core import Budget
from .raw_distillation import check_weights
from .registered import registered


def build(spec):
    text, tok, _ = load_base()
    last = len(text.layers) - 1
    quantize_prefix(text, last, 8)
    mx.random.seed(spec["seed"])
    decoder = tool_view(text, last, spec["rank"], spec["passes"])
    model = CopyReader(decoder, spec["pointer_width"])
    model.eval()
    return model, tok


def prepare(net, tok, rows, budget):
    features = []
    aligned = 0
    for row in rows:
        _, prefix, span, offsets, origin = request(
            tok, row["title"], row["passage"], row["question"]
        )
        assert fingerprint(prefix) == row["sha256"] and list(span) == row["source_span"]
        reply = tok.encode("</think>\n\n" + row["answer"], add_special_tokens=False) + [
            tok.convert_tokens_to_ids("<|im_end|>")
        ]
        # These publisher indices are used only for the auxiliary training loss.
        lo = row["answer_char"]
        answer = row["answer"]
        if row["unanswerable"] or row["passage"][lo : lo + len(answer)] != answer:
            lo = -1
        gold = [
            i
            for i, (a, b) in enumerate(offsets)
            if lo >= 0 and b > origin + lo and a < origin + lo + len(answer)
        ]
        aligned += bool(gold)
        features.append(
            (
                prefix + reply[:-1],
                len(prefix) - 1,
                reply,
                span,
                (min(gold), max(gold) + 1) if gold else (-1, -1),
            )
        )
    bank = [None] * len(rows)
    indices = sorted(range(len(rows)), key=lambda i: len(features[i][0]))
    for start in range(0, len(indices), 8):
        ix = indices[start : start + 8]
        width = max(len(features[i][0]) for i in ix)
        tokens = mx.array(
            [features[i][0] + [0] * (width - len(features[i][0])) for i in ix]
        )
        h = net.decoder.prefix_hidden(tokens)
        mx.eval(h)
        for j, i in enumerate(ix):
            bank[i] = h[j, : len(features[i][0])]
        budget.check()
    return bank, features, aligned


def fit(net, tok, spec, root, budget):
    train = read(spec["training"])["rows"]
    begun = perf_counter()
    bank, features, aligned = prepare(net, tok, train, budget)
    prep = perf_counter() - begun
    rng = random.Random(spec["seed"])
    groups = [
        [i for i, w in enumerate(train) if w["domain"] == d] for d in ("SQuAD", "QuAC")
    ]
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)

    def loss(m, h, ids, positions, y, source, annotation):
        states = m.decoder._execute(h, m.decoder.sites[m.decoder.span.start :])
        states = m.decoder.text.model.norm(states)
        ix = mx.arange(ids.shape[0])[:, None]
        current = states[ix, positions]
        vocab = m.decoder.project(current).astype(mx.float32)
        lv = mx.take_along_axis(vocab, y[..., None], -1)[..., 0] - mx.logsumexp(
            vocab, axis=-1
        )
        shift = mx.pad(ids[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
        axis = mx.arange(ids.shape[1])[None, :]
        mask = (axis >= source[:, 0:1]) & (axis < source[:, 1:2])
        prev = mx.where(axis > source[:, 0:1], shift, -1)
        attn, g = m.probabilities(
            current, m.source_keys(states), ids[ix, positions], prev, mask
        )
        matches = ids[:, None, :] == y[..., None]
        copy = (attn * matches).sum(-1)
        mix = (1 - g[..., 0]) * mx.exp(lv) + g[..., 0] * copy
        pointer_target = (
            matches
            & (axis[:, None, :] >= annotation[:, 0, None, None])
            & (axis[:, None, :] < annotation[:, 1, None, None])
        )
        align = (attn * pointer_target).sum(-1)
        eligible = pointer_target.any(-1)
        auxiliary = -mx.where(
            eligible, mx.log(mx.maximum(align, 1e-12)), 0
        ).sum() / mx.maximum(eligible.sum(), 1)
        return (
            -mx.log(mx.maximum(mix, 1e-30)).mean()
            - spec["language_weight"] * lv.mean()
            + spec["alignment_weight"] * auxiliary
        )

    grad = nn.value_and_grad(net, loss)
    metric = []
    for step in range(spec["updates"]):
        # Two independent source-balanced draws, each passage participating irrespective of length.
        ix = [i for group in groups for i in rng.sample(group, spec["batch"] // 2)]
        rng.shuffle(ix)
        width = max(len(features[i][0]) for i in ix)
        h = mx.stack([mx.pad(bank[i], ((0, width - len(bank[i])), (0, 0))) for i in ix])
        ids, ss, ys, source, annotation = [], [], [], [], []
        for i in ix:
            seq, first, reply, span, gold = features[i]
            selected = rng.choices(range(len(reply)), k=spec["targets"])
            ids.append(seq + [0] * (width - len(seq)))
            ss.append([first + j for j in selected])
            ys.append([reply[j] for j in selected])
            source.append(span)
            annotation.append(gold)
        cost, g = grad(net, h, *map(mx.array, (ids, ss, ys, source, annotation)))
        g, _ = optim.clip_grad_norm(g, 1)
        opt.update(net, g)
        mx.eval(cost, net.trainable_parameters(), opt.state)
        budget.check()
        if step % 100 == 0 or step + 1 == spec["updates"]:
            metric.append([step + 1, cost.item()])
            print("pointer update", metric[-1], flush=True)
    values = dict(tree_flatten(net.trainable_parameters()))
    if any(k.startswith("decoder.") and ".lora_" not in k for k in values):
        raise ValueError("Frozen Base weights appeared in the optimizer")
    path = root / "reader.safetensors"
    mx.save_safetensors(str(path), values)
    save(
        root / "fit.json",
        {
            "weight_sha256": digest(path),
            "training": digest(Path(spec["training"])),
            "parameters": sum(v.size for v in values.values()),
            "updates": spec["updates"],
            "documents": len(train),
            "aligned": aligned,
            "targets": spec["updates"] * spec["batch"] * spec["targets"],
            "prepare_seconds": prep,
            "optimizer_seconds": perf_counter() - begun - prep,
            "loss": metric,
        },
    )
    net.eval()


def score(tok, ids, w):
    scored = [grade(tok, ids, y) for y in w["references"]]
    return {
        "visible": scored[0]["visible"],
        "completed": scored[0]["completed"],
        "exact": any(v["exact"] for v in scored),
        "content": any(v["content"] for v in scored),
        "f1": max(v["f1"] for v in scored),
    }


def outcome(data, rows, spec):
    groups = {}
    for name in ("all", "SQuAD", "QuAC", "unanswerable"):
        index = [
            i
            for i, w in enumerate(rows)
            if name in ("all", w["domain"])
            or (name == "unanswerable" and w["unanswerable"])
        ]
        group = {}
        for side, arm in data.items():
            chosen = [arm[i] for i in index]
            group[side] = {
                k: sum(r["score"][k] for r in chosen)
                for k in ("completed", "content", "exact")
            }
            group[side].update(
                n=len(chosen),
                f1=sum(r["score"]["f1"] for r in chosen) / len(chosen),
                seconds=sum(r["seconds"] for r in chosen),
                ttft=sum(r["ttft"] for r in chosen),
                peak=max(r["peak"] for r in chosen),
            )
        x, y = (data[s] for s in ("candidate", "control"))
        changes = [
            int(x[i]["score"]["exact"]) - int(y[i]["score"]["exact"]) for i in index
        ]
        wins, losses = changes.count(1), changes.count(-1)
        group.update(gains=wins, losses=losses, p_exact=exact_mcnemar(wins, losses))
        groups[name] = group
    g = groups["all"]
    a, b = g["candidate"], g["control"]
    passed = (
        g["gains"] - g["losses"] >= spec["net_gain"]
        and g["p_exact"] <= 0.05
        and a["f1"] >= b["f1"] + spec["f1_gain"]
        and all(
            v["candidate"][k] >= v["control"][k]
            for v in groups.values()
            for k in ("content", "exact", "f1", "completed")
        )
        and all(a[k] <= b[k] for k in ("seconds", "ttft", "peak"))
    )
    return {"passed": bool(passed), "groups": groups}


def measure(net, tok, w, side, spec, budget):
    mx.synchronize()
    at = perf_counter()
    text, ids, span, _, _ = request(tok, w["title"], w["passage"], w["question"])
    assert ids == w["prompt_ids"] and fingerprint(ids) == w["sha256"]
    view = PassageRequest(net, span) if side == "candidate" else net
    first = None
    prompt_length = len(ids)

    def progress(consumed):
        nonlocal first
        if first is None and consumed == prompt_length:
            first = perf_counter() - at
        budget.check()

    r = generate(
        view,
        tok,
        text,
        spec["cap"],
        stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
        progress=progress,
    )
    if first is None:
        raise ValueError("Missing first emitted token timestamp")
    elapsed = perf_counter() - at
    ids = r["token_ids"]
    return {
        "id": w["id"],
        "input_sha256": w["sha256"],
        "ids": ids,
        "score": score(tok, ids, w),
        "seconds": elapsed,
        "ttft": first,
        "peak": r["peak_bytes"],
        "copy_gate": view.gates if side == "candidate" else [],
    }


def live_parity(net, rows, spec, budget):
    audit = []
    for i in spec["parity_indices"]:
        w = rows[i]
        ids = w["prompt_ids"]
        pair = []
        for width in (len(ids), 128):
            request_view = PassageRequest(net, w["source_span"])
            cache = request_view.make_cache()
            for start in range(0, len(ids), width):
                h = request_view.hidden(mx.array([ids[start : start + width]]), cache)
                mx.eval(h)
            lp = request_view.project(h[:, -1:])
            pair.append([int(mx.argmax(lp).item()), float(mx.max(lp).item())])
            budget.check()
        audit.append(
            {"index": i, "sha256": w["sha256"], "single": pair[0], "chunked": pair[1]}
        )
    return audit


def parity_passes(live, rows, spec):
    assert [r["index"] for r in live] == spec["parity_indices"]
    assert all(rows[r["index"]]["sha256"] == r["sha256"] for r in live)
    return all(
        r["single"][0] == r["chunked"][0]
        and abs(r["single"][1] - r["chunked"][1]) <= 0.04
        for r in live
    )


def verify(root, d):
    spec = read(root / "registration.json")
    rows = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    meta = read(root / "fit.json")
    assert meta["weight_sha256"] == digest(root / "reader.safetensors") and meta[
        "training"
    ] == digest(Path(spec["training"]))
    assert (
        meta["targets"] == spec["updates"] * spec["batch"] * spec["targets"]
        and meta["documents"] == 2048
    )
    data = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    for side, arm in data.items():
        for r, w in zip(arm, rows, strict=True):
            assert (
                r["id"] == w["id"]
                and r["input_sha256"] == fingerprint(w["prompt_ids"]) == w["sha256"]
            )
            assert (
                r["score"] == score(tok, r["ids"], w)
                and 0 < len(r["ids"]) <= spec["cap"]
            )
            assert 0 < r["ttft"] <= r["seconds"] and r["peak"] > 0
            assert len(r["copy_gate"]) == (len(r["ids"]) if side == "candidate" else 0)
            assert all(0 <= g <= 1 for g in r["copy_gate"])
    expected = outcome(data, rows, spec)
    expected["incremental"] = parity_passes(read(root / "live.json"), rows, spec)
    expected["passed"] = expected["passed"] and expected["incremental"]
    assert expected == d


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    rows = read(spec["material"])["rows"]
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                net, tok = build(spec)
                if (root / "fit.json").is_file():
                    meta = read(root / "fit.json")
                    path = root / "reader.safetensors"
                    if meta["weight_sha256"] != digest(path) or meta[
                        "training"
                    ] != digest(Path(spec["training"])):
                        raise ValueError("Mismatched reader fit")
                    check_weights(net, mx.load(str(path)))
                else:
                    fit(net, tok, spec, root, budget)
            else:
                net, tok = base()[:2]
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            arm = []
            for w in rows:
                arm.append(measure(net, tok, w, side, spec, budget))
                if len(arm) % 8 == 0:
                    print(
                        side,
                        len(arm),
                        sum(r["score"]["exact"] for r in arm),
                        flush=True,
                    )
            packed(root / (side + ".json.gz"), arm)
            data[side] = arm
            if side == "candidate":
                live = live_parity(net, rows, spec, budget)
                save(root / "live.json", live)
            del net, tok
            gc.collect()
            mx.clear_cache()
        decision = outcome(data, rows, spec)
        decision["incremental"] = parity_passes(live, rows, spec)
        decision["passed"] = decision["passed"] and decision["incremental"]
        save(root / "decision.json", decision)
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
