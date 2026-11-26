"""Fit and compare recurrent evidence reads on publisher-held-out two-hop stories."""

import argparse
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from .. import REVISION
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..evidence_loop import (
    EvidenceLoop,
    PooledEvidence,
    load_evidence,
    passage,
    pooled_states,
    prepare,
    sentence_states,
)
from ..io import digest, read, save
from . import core, deduction_data

PANEL = Path("research/data/deduction-stories-01.json")
REG = Path("research/experiments")
WORDS = ("cat", "mouse", "wolf", "sheep")


def load_material(s, freeze):
    panel = Path(s.get("material", PANEL))
    if freeze:
        if "pairs" in s:
            from .deduction_counterfactual import material

            rows = material(s)
        else:
            rows = deduction_data.material(s)
        if panel.exists() and read(panel) != rows:
            raise ValueError("Frozen stories differ")
        if not panel.exists():
            save(panel, rows)
        return rows
    idx = read("research/registry.json")
    if idx["datasets"][str(panel)]["sha256"] != digest(panel) or idx["experiments"][
        s["id"]
    ]["registration"] != digest(REG / (s["id"] + ".json")):
        raise ValueError("Unregistered material")
    return read(panel)


def training_inputs(
    model, t, rows, depth, b, encoder, response=deduction_data.response
):
    responses = [
        t.encode(response(r["answer"]), add_special_tokens=False) for r in rows
    ]
    vocab = sorted(set().union(*map(set, responses)))
    if len({len(r) for r in responses}) != 1:
        raise ValueError("Different target lengths require masking")
    texts = sorted({text for r in rows for text in [*r["facts"], r["question"]]})
    features = []
    for i in range(0, len(texts), 16):
        b.check()
        pair = encoder(model, t, texts[i : i + 16], depth)
        mx.eval(*pair)
        features.append(pair)
    first, second = [mx.concatenate(frags) for frags in zip(*features)]
    lookup = {text: i for i, text in enumerate(texts)}
    inputs = []
    for r, ids in zip(rows, responses, strict=True):
        b.check()
        x = mx.array([r["prompt_ids"] + ids[:-1]])
        h = model.hidden(x)[:, -len(ids) :]
        z = model.project(h).astype(mx.float32)
        # All vocabulary mass outside the fitted residual columns stays at its
        # original logit. This computes the exact full-vocabulary denominator.
        other = z.at[..., mx.array(vocab)].add(-float("inf"))
        values = [h[0], z[0, :, mx.array(vocab)].T, mx.logsumexp(other[0], axis=-1)]
        if values[1].shape != (len(ids), len(vocab)):
            raise ValueError("Incorrect vocabulary projection layout")
        mx.eval(*values)
        inputs.append(values)
    h, z, other = [mx.stack(parts) for parts in zip(*inputs)]
    facts = mx.array([[lookup[f] for f in r["facts"]] for r in rows])
    question = mx.array([lookup[r["question"]] for r in rows])
    targets = mx.array([[vocab.index(tok) for tok in ids] for ids in responses])
    support = mx.array([r["supports"] for r in rows])
    return (
        vocab,
        (first, second),
        (
            first[facts],
            second[facts],
            first[question],
            h,
            z,
            other,
            targets,
            support,
        ),
    )


def fit(model, t, rows, depth, s, b, out, span_passes):
    pooled = s.get("architecture") == "pooled"
    vocab, moments, arrays = training_inputs(
        model, t, rows, span_passes, b, pooled_states if pooled else sentence_states
    )
    mx.random.seed(s["fit_seed"])
    net = (PooledEvidence if pooled else EvidenceLoop)(
        model.args.hidden_size, vocab, s["rank"]
    )
    if pooled:
        net.calibrate(*moments)
    word_columns = mx.array([vocab.index(core.atom(t, " " + w)) for w in WORDS])
    opt = optim.Adam(s["lr"])

    def loss(net, facts, deep, q, h, z, other, y, support):
        mem, p, edges, _ = net.read(facts, deep, q, depth)
        logits = z + net.bias(h, mem)
        logz = mx.logaddexp(mx.logsumexp(logits, axis=-1), other)
        per_token = logz - mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        weights = 1 + s.get("lexical_weight", 0) * mx.any(
            y[..., None] == word_columns, axis=-1
        )
        ce = mx.sum(per_token * weights) / mx.sum(weights)
        ix = mx.arange(len(q))
        aux = -mx.mean(
            mx.log(p[ix, support[:, 0]] + 1e-12)
            + mx.log(edges[ix, support[:, 0], support[:, 1]] + 1e-12)
        )
        return ce + s["support_weight"] * aux

    grad = nn.value_and_grad(net, loss)
    rng = random.Random(s["fit_seed"])
    steps = []
    training_time = time.perf_counter()
    for step in range(s["updates"]):
        b.check()
        pick = mx.array(rng.sample(range(len(rows)), s["batch"]))
        val, g = grad(net, *[x[pick] for x in arrays])
        opt.update(net, g)
        mx.eval(net.parameters(), opt.state, val)
        if step % 100 == 0 or step + 1 == s["updates"]:
            steps.append([step + 1, val.item()])
    net.save_weights(str(out / "reader.safetensors"))
    spec = {
        "base_revision": REVISION,
        "architecture": s.get("architecture", "terminal"),
        "span_passes": span_passes,
        "depth": depth,
        "rank": s["rank"],
        "vocab": vocab,
        "weights": digest(out / "reader.safetensors"),
        "updates": s["updates"],
        "loss": steps,
        "optimization_seconds": time.perf_counter() - training_time,
    }
    save(out / "reader.json", spec)
    return net


def evaluate(model, t, net, rows, depth, s, b, path, span_passes):
    ids = [core.atom(t, " " + w) for w in WORDS]
    probe_suffix = t.encode(
        "After considering the evidence, I conclude.\n</think>\n\nThe animal is a",
        add_special_tokens=False,
    )
    records = []
    for r in rows:
        b.check()
        start = time.perf_counter()
        view = model
        attn = []
        if depth:
            parts = passage(t, r["prompt_ids"])
            if parts != (r["facts"], r["question"]):
                raise ValueError("Passage differs from the user tokens")
            view, first, last = prepare(
                model, t, net, *parts, depth, span_passes=span_passes
            )
            attn = [first, last]
        v = core.generate(
            view, t, r["prompt_ids"], s["generation_cap"], b.check, thinking=True
        )
        elapsed = time.perf_counter() - start
        z = view.project(view.hidden(mx.array([r["prompt_ids"] + probe_suffix]))[:, -1])
        logits = z[0, mx.array(ids)].tolist()
        pred = deduction_data.answer(v["visible"]) if v["complete"] else None
        records.append(
            dict(
                **v,
                id=r["id"],
                answer=pred,
                correct=pred == r["answer"],
                probe=WORDS[max(range(len(WORDS)), key=logits.__getitem__)],
                probe_logits=logits,
                attention=attn,
                end_to_end=elapsed,
            )
        )
        packed(path, {"records": records})
    return records


def summary(rows, records):
    return {
        "correct": sum(r["correct"] for r in records),
        "complete": sum(r["complete"] for r in records),
        "probe": sum(
            r["probe"] == x["answer"] for x, r in zip(rows, records, strict=True)
        ),
        "seconds": sum(r["end_to_end"] for r in records),
        "tokens": sum(len(r["token_ids"]) for r in records),
    }


def run(name, freeze=False):
    s = read(REG / (name + ".json"))
    data = load_material(s, freeze)
    if freeze:
        return
    prior = s.get("requires")
    if prior and (
        not read(Path("research/results") / prior / "decision.json")["passed"]
        or digest(Path("research/results") / prior / "checksums.json")
        != s["prerequisite_sha256"]
    ):
        raise ValueError("Previous paired gate failed")
    b = core.Budget(name, s["limits"])
    out = core.OUT / name
    out.mkdir(exist_ok=False, parents=True)
    res = {
        "registration": digest(REG / (name + ".json")),
        "data": digest(Path(s.get("material", PANEL))),
        "passed": False,
    }
    try:
        model, t, _ = base()
        rows = data[s["panel"]]
        results = {}
        for arm, spec in s["arms"].items():
            depth = spec["depth"]
            span_passes = spec.get("span_passes", depth)
            net = None
            if spec.get("fit"):
                folder = (
                    out / arm
                    if sum(bool(v.get("fit")) for v in s["arms"].values()) > 1
                    else out
                )
                folder.mkdir(exist_ok=True)
                net = fit(model, t, data["train"], depth, s, b, folder, span_passes)
            elif depth:
                folder = Path(spec["reader"])
                if digest(folder / "reader.safetensors") != s["reader_weights"][arm]:
                    raise ValueError("Changed preregistered reader")
                net = load_evidence(model, folder)
            records = evaluate(
                model, t, net, rows, depth, s, b, out / (arm + ".json.gz"), span_passes
            )
            results[arm] = summary(rows, records)
        paths = [unpack(out / (k + ".json.gz"))["records"] for k in s["arms"]]
        delta = [
            int(x["correct"]) - int(y["correct"]) for x, y in zip(*paths, strict=True)
        ]
        a, c = results.values()
        wins, losses = delta.count(1), delta.count(-1)
        p = exact_mcnemar(wins, losses)
        res.update(
            status="completed",
            scores=results,
            wins=wins,
            losses=losses,
            p=p,
            passed=sum(delta) >= s["net_floor"]
            and p < 0.05
            and a["probe"] >= c["probe"] + s["probe_floor"]
            and a["seconds"] <= c["seconds"] * s["time_ratio"],
        )
        if "pairs" in s:
            pair_counts = {}
            for arm, records in zip(s["arms"], paths, strict=True):
                pairs = {}
                for item, record in zip(rows, records, strict=True):
                    pairs.setdefault(item["pair"], []).append(record["correct"])
                if len(pairs) != s["pairs"] or any(len(v) != 2 for v in pairs.values()):
                    raise ValueError("Incomplete pair inventory")
                pair_counts[arm] = sum(all(v) for v in pairs.values())
            a, b_control = pair_counts.values()
            res["both_correct"] = pair_counts
            res["passed"] &= a >= s["pair_floor"] and a - b_control >= s["pair_net"]
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        res.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(out / "decision.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    run(args.name, args.freeze)
