"""Transfer a native training teacher's inline answer distributions to both input views."""

import gc
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from .. import checkpoint
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..model_loader import count
from .file_evidence_fit import pack, projected
from .native_function_data import token_pair


def answer_positions(tok, ids, answer):
    close = ids.index(tok.convert_tokens_to_ids("</think>"))
    start = close + 1
    while start < len(ids) - 1 and not tok.decode([ids[start]]).strip():
        start += 1
    assert tok.decode(ids[start:-1]).strip() == answer.strip()
    return list(range(start, len(ids) - 1))


def teacher(spec, tok, root, budget):
    path = root / "teacher.json.gz"
    out = root / "teacher-summary.json"
    if path.is_file() and out.is_file():
        summary = read(out)
        assert (
            digest(path) == summary["targets_sha256"]
            and summary["first_gold_top_k_rate"] >= spec["min_teacher_coverage"]
        )
        return
    cfg = spec["teacher"]
    directory = Path(cfg["directory"])
    assert digest(directory / "model.safetensors") == cfg["weights_sha256"]
    assert all(digest(directory / k) == v for k, v in cfg["files"].items())
    assert (
        Tokenizer.from_file(str(directory / "tokenizer.json")).get_vocab()
        == tok.get_vocab()
    )
    started = perf_counter()
    model, _ = load_model(directory)
    model.eval()
    text = model.language_model
    records = {}
    hits = targets = 0
    temp = spec["temperature"]
    k = spec["top_k"]
    for row in read(spec["training"])["examples"]:
        if row["kind"] != "inline":
            continue
        budget.check()
        prefix, ids = token_pair(tok, row)
        pos = answer_positions(tok, ids, row["target"]["content"])
        sequence = mx.array([prefix + ids[:-1]])
        hidden = text.model(sequence)[0, mx.array([len(prefix) - 1 + j for j in pos])]
        logits = (
            text.model.embed_tokens.as_linear(hidden)
            if text.args.tie_word_embeddings
            else text.lm_head(hidden)
        ).astype(mx.float32)
        probs = mx.softmax(logits / temp, axis=-1)
        top = mx.argpartition(-probs, kth=k - 1, axis=-1)[:, :k]
        values = mx.take_along_axis(probs, top, axis=-1)
        order = mx.argsort(-values, axis=-1)
        top = mx.take_along_axis(top, order, axis=-1)
        values = mx.take_along_axis(values, order, axis=-1)
        mx.eval(top, values)
        top, values = top.tolist(), values.tolist()
        hits += ids[pos[0]] in top[0]
        targets += len(pos)
        records[row["source_question"]] = {
            "positions": pos,
            "labels": [ids[j] for j in pos],
            "ids": top,
            "probabilities": values,
        }
        if len(records) % 64 == 0:
            print("native teacher", len(records), flush=True)
    rate = hits / len(records)
    assert len(records) == 2 * spec["train_articles"]
    packed(path, records)
    save(
        out,
        {
            "teacher_weights": cfg["weights_sha256"],
            "training_sha256": digest(Path(spec["training"])),
            "targets_sha256": digest(path),
            "examples": len(records),
            "answer_positions": targets,
            "first_gold_top_k": hits,
            "first_gold_top_k_rate": rate,
            "seconds": perf_counter() - started,
        },
    )
    if rate < spec["min_teacher_coverage"]:
        raise InterruptedError(
            "Training teacher did not reach preregistered first-answer coverage"
        )
    del model, text
    gc.collect()
    mx.clear_cache()


def fit(model, tok, s, root, budget, *, prepare):
    start = perf_counter()
    training = read(s["training"])
    examples = training["examples"]
    bank = prepare(model, tok, training, budget)
    targets = unpack(root / "teacher.json.gz")
    articles = defaultdict(dict)
    rehearsal = []
    for row, sample in zip(examples, bank, strict=True):
        if row["kind"] == "rehearsal":
            rehearsal.append(sample)
            continue
        question = articles[row["source_article"]].setdefault(
            row["source_question"], {}
        )
        question[row["kind"]] = sample
    assert all(
        len(qs) == 2
        and all(set(views) == {"inline", "file", "opening"} for views in qs.values())
        for qs in articles.values()
    )
    groups = list(articles.values())
    model.train()
    model.checkpointing = True
    rng = random.Random(s["fit_seed"])
    optimizer = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    logs = []
    tokens_seen = positions_seen = 0
    previous = 0
    prior_time = 0.0
    preptime = perf_counter() - start
    temp = s["temperature"]

    def generic(net, h, r, c, labels):
        return nn.losses.cross_entropy(projected(net, h, r, c), labels).mean()

    def reading(net, h, r, c, labels, indices, ids, prob):
        logits = projected(net, h, r, c)
        exact = nn.losses.cross_entropy(logits[indices], labels[indices]).mean()
        formatting = nn.losses.cross_entropy(logits, labels).mean()
        chosen = logits[indices] / temp
        logq = chosen - mx.logsumexp(chosen, axis=-1, keepdims=True)
        values = mx.take_along_axis(logq, ids, axis=-1)
        tail = mx.maximum(1.0 - mx.exp(values).sum(-1), 1e-8)
        remainder = mx.maximum(1.0 - prob.sum(-1), 0.0)
        distill = -(mx.sum(prob * values, axis=-1) + remainder * mx.log(tail)).mean()
        return exact + 0.2 * formatting + 0.45 * temp**2 * distill

    natural = nn.value_and_grad(model, reading)
    ordinary = nn.value_and_grad(model, generic)
    directory = Path("checkpoints/research") / s["id"]
    directory.mkdir(parents=True, exist_ok=True)
    binding = digest(root / "registration.json")

    def draw():
        qs = rng.choice(groups)
        return qs, rng.choice(list(qs.values()))["opening"], rng.choice(rehearsal)

    if (directory / "latest.json").is_file():
        state = checkpoint.restore(directory, model, optimizer)
        assert state["registration"] == binding
        logs = state["loss"]
        tokens_seen, positions_seen = state["tokens"], state["positions"]
        previous, prior_time = state["step"], state["seconds"]
        for _ in range(previous):
            draw()
    start = perf_counter()
    for step in range(previous, s["updates"]):
        budget.check()
        qs, opening, other = draw()
        grads = None
        value = 0.0
        for key, v in qs.items():
            a, b = v["inline"], v["file"]
            assert a[2] == b[2]
            target = targets[key]
            pos = target["positions"]
            assert [a[2][j] for j in pos] == target["labels"]
            ix = mx.array([*pos, *(len(a[2]) + j for j in pos)])
            ids = mx.array(target["ids"] * 2)
            probs = mx.array(target["probabilities"] * 2)
            loss, g = natural(model, *pack([a, b]), ix, ids, probs)
            g = tree_map(lambda t: t.astype(mx.float32) * 0.3, g)
            grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
            mx.eval(loss, grads)
            value += 0.3 * loss.item()
            tokens_seen += len(a[2]) + len(b[2])
            positions_seen += 2 * len(pos)
        for row, weight in ((opening, 0.30), (other, 0.10)):
            loss, g = ordinary(model, *pack([row]))
            grads = tree_map(lambda x, y: x + weight * y.astype(mx.float32), grads, g)
            mx.eval(loss, grads)
            value += weight * loss.item()
            tokens_seen += len(row[2])
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        budget.check()
        if step == 0 or (step + 1) % 80 == 0 or step + 1 == s["updates"]:
            logs.append({"step": step + 1, "loss": value})
            print("cross-view fit", logs[-1], flush=True)
            if step:
                checkpoint.write(
                    directory,
                    model,
                    optimizer,
                    {
                        "step": step + 1,
                        "registration": binding,
                        "loss": logs,
                        "tokens": tokens_seen,
                        "positions": positions_seen,
                        "seconds": prior_time + perf_counter() - start,
                    },
                )
    weights = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(model.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "passes": 2,
            "weights": digest(weights),
            "training": digest(Path(s["training"])),
            "updates": s["updates"],
            "turns": len(bank),
            "targets": tokens_seen,
            "teacher_positions": positions_seen,
            "trainable": count(model, True),
            "prepare_seconds": preptime,
            "optimizer_seconds": prior_time + perf_counter() - start,
            "loss": logs,
        },
    )
