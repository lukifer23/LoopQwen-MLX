"""Train on human-judged Wikipedia statements and test semantic transfer to real technical support."""

import argparse
import gc
import random
from collections import defaultdict
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
from ..evaluation.relations import conclusion
from ..evaluation.statistics import exact_mcnemar
from ..identity import fingerprint
from ..io import digest, read, save
from ..meaning_alignment import AlignedReader, PairRequest, request
from ..model_loader import load_base, local_snapshot
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
    model = AlignedReader(
        tool_view(text, last, spec["rank"], spec["passes"]), spec["width"]
    )
    model.eval()
    return model, tok


def encode(tok, r, instructions):
    text, ids, spans = request(tok, r["left"], r["right"], instructions[r["domain"]])
    assert fingerprint(ids) == r["sha256"] and spans == r["spans"]
    return text, ids, spans


def replies(tok):
    close = tok.encode("</think>\n\n", add_special_tokens=False)
    labels = [tok.encode(w, add_special_tokens=False) for w in ("No", "Yes")]
    assert all(len(x) == 1 for x in labels)
    tokens = [v[0] for v in labels]
    assert all(
        tok.encode("</think>\n\n" + s, add_special_tokens=False) == close + [i]
        for s, i in zip(("No", "Yes"), tokens, strict=True)
    )
    return close, tokens, tok.convert_tokens_to_ids("<|im_end|>")


def fit(model, tok, spec, out, budget):
    enrolled = read(spec["training"])
    rows = enrolled["rows"]
    close, labels, eos = replies(tok)
    ids = []
    replies_by_item = []
    for r in rows:
        _, prefix, _ = encode(tok, r, enrolled["instructions"])
        reply = close + [labels[r["label"]], eos]
        ids.append(prefix + reply[:-1])
        replies_by_item.append(reply)
    bank = [None] * len(rows)
    indices = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    begin = perf_counter()
    for start in range(0, len(indices), 12):
        ix = indices[start : start + 12]
        width = max(len(ids[i]) for i in ix)
        h = model.decoder.prefix_hidden(
            mx.array([ids[i] + [0] * (width - len(ids[i])) for i in ix])
        )
        mx.eval(h)
        for j, i in enumerate(ix):
            bank[i] = h[j, : len(ids[i])]
        budget.check()
    prep = perf_counter() - begin

    def stacked(values):
        width = max(a.shape[0] for a in values)
        return mx.stack(
            [mx.pad(a, ((0, width - a.shape[0]), (0, 0))) for a in values]
        ), mx.arange(width)[None, :] < mx.array([a.shape[0] for a in values])[:, None]

    def loss(net, h, left, right, lm, rm, positions, gold):
        rel = net.relation(left, right, lm, rm)
        start = positions[:, 0]
        h = net.condition(h, rel, mx.arange(h.shape[1])[None, :] >= start[:, None])
        y = net.read_suffix(h)[mx.arange(h.shape[0])[:, None], positions]
        logits = net.decoder.project(y).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, gold, reduction="mean")
        yn = logits[:, len(close), labels[1]] - logits[:, len(close), labels[0]]
        # Adjacent items share their first text, and carry opposite judgments.
        return (
            ce + spec["pair_weight"] * mx.logaddexp(0, 1 - (yn[1::2] - yn[0::2])).mean()
        )

    grad = nn.value_and_grad(model, loss)
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    rng = random.Random(spec["seed"])
    groups = list(range(len(rows) // 2))
    assert all(
        rows[2 * i]["pair"] == rows[2 * i + 1]["pair"]
        and [rows[2 * i + j]["label"] for j in (0, 1)] == [0, 1]
        for i in groups
    )
    per = spec["batch"] // 2
    assert len(groups) % per == 0
    history = []
    for step in range(spec["updates"]):
        ix0 = step * per % len(groups)
        if ix0 == 0:
            rng.shuffle(groups)
        ix = [2 * g + j for g in groups[ix0 : ix0 + per] for j in (0, 1)]
        h, _ = stacked([bank[i] for i in ix])

        def side(j):
            slices = [rows[i]["spans"][j] for i in ix]
            return stacked([bank[i][a:b] for i, (a, b) in zip(ix, slices, strict=True)])

        left, lm = side(0)
        right, rm = side(1)
        y = mx.array([replies_by_item[i] for i in ix])
        positions = (
            mx.array([len(ids[i]) - len(replies_by_item[i]) for i in ix])[:, None]
            + mx.arange(y.shape[1])[None, :]
        )
        cost, gr = grad(model, h, left, right, lm, rm, positions, y)
        gr, _ = optim.clip_grad_norm(gr, 1)
        opt.update(model, gr)
        mx.eval(cost, model.trainable_parameters(), opt.state)
        budget.check()
        if step % 100 == 0 or step + 1 == spec["updates"]:
            history.append([step + 1, cost.item()])
            print("alignment update", history[-1], flush=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    if any(k.startswith("decoder.") and ".lora_" not in k for k in weights):
        raise ValueError("Base entered the optimizer")
    dest = out / "alignment.safetensors"
    mx.save_safetensors(str(dest), weights)
    save(
        out / "fit.json",
        dict(
            updates=spec["updates"],
            examples=len(rows),
            unique_pairs=len(groups),
            targets=spec["updates"] * spec["batch"] * len(replies_by_item[0]),
            parameters=sum(v.size for v in weights.values()),
            weights=digest(dest),
            training=digest(Path(spec["training"])),
            prepare_seconds=prep,
            optimizer_seconds=perf_counter() - begin - prep,
            loss=history,
        ),
    )
    model.eval()


def score(tok, ids, label):
    s = grade(tok, ids, ("No", "Yes")[label])
    pred = conclusion(s["visible"])
    return dict(
        visible=s["visible"],
        completed=s["completed"],
        judgment=pred,
        correct=pred == ("no", "yes")[label],
    )


def probe(view, ids, labels, block):
    cache = view.make_cache()
    for i in range(0, len(ids), block):
        h = view.hidden(mx.array([ids[i : i + block]]), cache)
        mx.eval(h)
    logits = view.project(h[:, -1:])[0, -1]
    a, b = map(float, logits[labels].astype(mx.float32).tolist())
    if a == b:
        return [a, b, None]
    return [a, b, int(b > a)]


def measure(net, tok, r, prompts, side, spec, budget, parity):
    mx.synchronize()
    at = perf_counter()
    text, ids, spans = encode(tok, r, prompts)
    assert ids == r["prompt_ids"]

    def view():
        return PairRequest(net, spans, len(ids)) if side == "candidate" else net

    _, _, eos = replies(tok)
    first = None

    def progress(consumed):
        nonlocal first
        if consumed == len(ids) and first is None:
            first = perf_counter() - at
        budget.check()

    output = generate(view(), tok, text, spec["cap"], stop_ids=[eos], progress=progress)
    seconds = perf_counter() - at
    assert first is not None
    close, labels, _ = replies(tok)
    prepared = ids + close
    t = perf_counter()
    logits = probe(view(), prepared, labels, 128)
    probe_s = perf_counter() - t
    whole = (
        probe(view(), prepared, labels, len(prepared))
        if parity and side == "candidate"
        else None
    )
    return dict(
        id=r["id"],
        sha256=fingerprint(ids),
        ids=output["token_ids"],
        score=score(tok, output["token_ids"], r["label"]),
        seconds=seconds,
        ttft=first,
        peak=output["peak_bytes"],
        probe=logits,
        whole=whole,
        probe_seconds=probe_s,
    )


def outcome(data, rows, spec):
    groups = {}
    for name in ("all", "Wikipedia", "AskUbuntu"):
        by = defaultdict(list)
        for i, r in enumerate(rows):
            if name in ("all", r["domain"]):
                by[r["pair"]].append(i)
        assert all(
            len(ix) == 2 and {rows[i]["label"] for i in ix} == {0, 1}
            for ix in by.values()
        )
        record = {}
        pair_order = {}
        for side, arm in data.items():
            indices = [i for ix in by.values() for i in ix]

            def hit(i):
                return arm[i]["score"]["correct"]

            def choice(i):
                return arm[i]["probe"][2] == rows[i]["label"]

            pairs = [all(hit(i) for i in ix) for ix in by.values()]
            pair_order[side] = pairs
            record[side] = dict(
                n=len(indices),
                groups=len(by),
                pairs=sum(pairs),
                probe_pairs=sum(all(choice(i) for i in ix) for ix in by.values()),
                positive=sum(hit(i) for i in indices if rows[i]["label"]),
                negative=sum(hit(i) for i in indices if not rows[i]["label"]),
                correct=sum(hit(i) for i in indices),
                probe=sum(choice(i) for i in indices),
                completed=sum(arm[i]["score"]["completed"] for i in indices),
                seconds=sum(arm[i]["seconds"] for i in indices),
                ttft=sum(arm[i]["ttft"] for i in indices),
                probe_seconds=sum(arm[i]["probe_seconds"] for i in indices),
                peak=max(arm[i]["peak"] for i in indices),
            )
        delta = [
            int(a) - int(b)
            for a, b in zip(pair_order["candidate"], pair_order["control"], strict=True)
        ]
        gains, losses = delta.count(1), delta.count(-1)
        record.update(gains=gains, losses=losses, p_exact=exact_mcnemar(gains, losses))
        groups[name] = record
    all_ = groups["all"]
    a, b = (all_[s] for s in ("candidate", "control"))
    cases = [r for r in data["candidate"] if r["whole"] is not None]
    assert len(cases) == len(spec["parity_indices"])
    parity = all(
        r["probe"][2] == r["whole"][2]
        and abs((r["whole"][1] - r["whole"][0]) - (r["probe"][1] - r["probe"][0]))
        <= spec["margin_tolerance"]
        for r in cases
    )
    passed = (
        all_["gains"] - all_["losses"] >= spec["net_pairs"]
        and all_["p_exact"] <= 0.05
        and a["probe_pairs"] >= b["probe_pairs"] + 2
        and all(
            v["candidate"]["pairs"] >= v["control"]["pairs"] + 2
            and all(
                v["candidate"][k] >= v["control"][k]
                for k in ("positive", "negative", "probe", "probe_pairs", "completed")
            )
            for v in groups.values()
        )
        and all(a[k] <= b[k] for k in ("seconds", "ttft", "peak"))
        and parity
    )
    return dict(passed=bool(passed), incremental=parity, groups=groups)


def verify(root, d):
    spec = read(root / "registration.json")
    rows = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    meta = read(root / "fit.json")
    assert (
        meta["weights"] == digest(root / "alignment.safetensors")
        and meta["training"] == digest(Path(spec["training"]))
        and meta["updates"] == spec["updates"]
    )
    data = {}
    for s in ("candidate", "control"):
        arm = unpack(root / (s + ".json.gz"))
        for i, (r, w) in enumerate(zip(arm, rows, strict=True)):
            assert (
                r["id"] == w["id"]
                and r["sha256"] == fingerprint(w["prompt_ids"]) == w["sha256"]
            )
            assert (
                r["score"] == score(tok, r["ids"], w["label"])
                and 0 < len(r["ids"]) <= spec["cap"]
            )
            assert (
                r["seconds"] >= r["ttft"] > 0
                and r["probe_seconds"] > 0
                and r["peak"] > 0
            )
            assert (r["whole"] is not None) == (
                s == "candidate" and i in spec["parity_indices"]
            )
            for logits in (r["probe"], r["whole"]):
                if logits:
                    assert logits[2] == (
                        None if logits[0] == logits[1] else int(logits[1] > logits[0])
                    )
        data[s] = arm
    assert d == outcome(data, rows, spec)


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decision.json").is_file():
        raise FileExistsError(out)
    panel = read(spec["material"])
    b = Budget(name, spec["limits"])
    data = {}
    try:
        for s in ("candidate", "control"):
            if s == "candidate":
                model, tok = build(spec)
                f = out / "alignment.safetensors"
                if (out / "fit.json").is_file():
                    meta = read(out / "fit.json")
                    assert meta["weights"] == digest(f) and meta["training"] == digest(
                        Path(spec["training"])
                    )
                    check_weights(model, mx.load(str(f)))
                else:
                    fit(model, tok, spec, out, b)
            else:
                model, tok = base()[:2]
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            rows = []
            for i, w in enumerate(panel["rows"]):
                rows.append(
                    measure(
                        model,
                        tok,
                        w,
                        panel["instructions"],
                        s,
                        spec,
                        b,
                        i in spec["parity_indices"],
                    )
                )
                if len(rows) % 8 == 0:
                    print(
                        s,
                        len(rows),
                        sum(r["score"]["correct"] for r in rows),
                        flush=True,
                    )
            packed(out / (s + ".json.gz"), rows)
            data[s] = rows
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(out / "decision.json", outcome(data, panel["rows"], spec))
    finally:
        b.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
