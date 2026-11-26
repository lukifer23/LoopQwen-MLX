"""Test learned two-fact evidence composition on human-authored science questions."""

import argparse
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from .. import REVISION
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..evaluation.statistics import exact_mcnemar
from ..evidence_loop import EvidenceView, pooled_states, unit
from ..io import digest, read, save
from . import core, science_data

PANEL = Path("research/data/science-composition-01.json")


class ScienceReader(nn.Module):
    def __init__(self, width, rank, vocab, letters):
        super().__init__()
        self.fact_keys = nn.Linear(4 * width, rank)
        self.queries = nn.Linear(4 * width, rank)
        self.value = nn.Linear(4 * width, rank)
        self.language = nn.Linear(width, rank)
        self.format = nn.Linear(rank, len(vocab))
        self.gate = nn.Linear(rank, 1)
        self.vocab = tuple(vocab)
        self.letters = tuple(vocab.index(x) for x in letters)
        self.first_mean = mx.zeros(4 * width)
        self.first_scale = mx.ones(4 * width)
        self.deep_mean = mx.zeros(4 * width)
        self.deep_scale = mx.ones(4 * width)

    def calibrate(self, a, b):
        for name, x in (("first", a), ("deep", b)):
            setattr(self, name + "_mean", x.mean(0))
            setattr(self, name + "_scale", mx.rsqrt(mx.var(x, axis=0) + 0.0001))
        self.freeze(keys=["deep_mean", "deep_scale", "first_mean", "first_scale"])

    def composition(self, f, deep, q, options, depth):
        f = (f - self.first_mean) * self.first_scale
        q = (q - self.first_mean) * self.first_scale
        o = (options - self.first_mean) * self.first_scale
        deep = (deep - self.deep_mean) * self.deep_scale
        k = unit(self.fact_keys(f))

        def relevance(x):
            return (unit(self.queries(x)) @ k.swapaxes(-1, -2)) * (10 / k.shape[-1])

        first = mx.softmax(relevance(q[:, None, :]), axis=-1)
        # The two distinct supporting facts may appear in either order.
        edge = mx.softmax(relevance(deep) - 1e4 * mx.eye(f.shape[1]), axis=-1)
        last = first @ edge if depth == 2 else first
        raw = (unit(f) @ unit(o).swapaxes(-1, -2)) / f.shape[-1]
        learned = unit(self.value(f)) @ unit(self.value(o)).swapaxes(-1, -2)
        affinity = 6 * raw + 10 * learned / k.shape[-1]
        scores = mx.logsumexp(mx.log(last.swapaxes(-1, -2) + 1e-12) + affinity, axis=1)
        return scores, first[:, 0], edge

    def bias(self, h, scores):
        hx = mx.tanh(self.language(unit(h)))
        residual = self.format(hx)
        return residual.at[..., mx.array(self.letters)].add(
            3 * mx.sigmoid(self.gate(hx)) * scores[:, None, :]
        )


def target(letter):
    return "I choose " + letter + ".\n</think>\n\nAnswer: " + letter + ".<|im_end|>"


def prep(model, t, rows, depth, s, b):
    vals = [t.encode(target(r["answer"]), add_special_tokens=False) for r in rows]
    vocab = sorted(set().union(*map(set, vals)))
    letters = [core.atom(t, " " + w) for w in science_data.LETTERS]
    texts = sorted(
        {x for r in rows for x in [*r["facts"], r["question"], *r["options"]]},
        key=lambda x: (len(x), x),
    )
    ix = {x: i for i, x in enumerate(texts)}
    features = []
    for i in range(0, len(texts), 12):
        b.check()
        a, deep = pooled_states(model, t, texts[i : i + 12], depth)
        a, deep = a.astype(mx.bfloat16), deep.astype(mx.bfloat16)
        mx.eval(a, deep)
        features.append((a, deep))
    a, deep = [mx.concatenate(v) for v in zip(*features)]
    hzs = []
    for r, ids in zip(rows, vals, strict=True):
        b.check()
        h = model.hidden(mx.array([r["prompt_ids"] + ids[:-1]]))[:, -len(ids) :]
        z = model.project(h).astype(mx.float32)
        rest = mx.logsumexp(z.at[..., mx.array(vocab)].add(-float("inf")), axis=-1)
        v = (h[0], mx.take(z[0], mx.array(vocab), axis=-1), rest[0])
        mx.eval(*v)
        hzs.append(v)
    arrays = [
        mx.array([[ix[x] for x in r["facts"]] for r in rows]),
        mx.array([ix[r["question"]] for r in rows]),
        mx.array([[ix[x] for x in r["options"]] for r in rows]),
        *[mx.stack(v) for v in zip(*hzs)],
        mx.array([[vocab.index(x) for x in row] for row in vals]),
        mx.array([r["supports"] for r in rows]),
        mx.array([science_data.LETTERS.index(r["answer"]) for r in rows]),
    ]
    return a, deep, arrays, vocab, letters


def fit(model, t, rows, depth, s, b, out):
    a, deep, arrays, vocab, letters = prep(model, t, rows, depth, s, b)
    mx.random.seed(s["fit_seed"])
    net = ScienceReader(model.args.hidden_size, s["rank"], vocab, letters)
    net.calibrate(a.astype(mx.float32), deep.astype(mx.float32))
    rng = random.Random(s["fit_seed"])
    opt = optim.Adam(s["lr"])

    def loss(net, fi, qi, oi, h, z, rest, tokens, sup, y):
        scores, p, edges = net.composition(a[fi], deep[fi], a[qi], a[oi], depth)
        idx = mx.arange(len(y))
        u, v = sup[:, 0], sup[:, 1]
        edge_loss = -mx.mean(
            mx.log(p[idx, u] * edges[idx, u, v] + p[idx, v] * edges[idx, v, u] + 1e-12)
        )
        z = z + net.bias(h, scores)
        nll = (
            mx.logaddexp(mx.logsumexp(z, axis=-1), rest)
            - mx.take_along_axis(z, tokens[..., None], axis=-1)[..., 0]
        )
        return (
            mx.mean(nll)
            + nn.losses.cross_entropy(scores, y, reduction="mean")
            + 0.35 * edge_loss
        )

    grad = nn.value_and_grad(net, loss)
    losses = []
    for step in range(s["updates"]):
        b.check()
        idx = mx.array(rng.sample(range(len(rows)), s["batch"]))
        v, g = grad(net, *[x[idx] for x in arrays])
        opt.update(net, g)
        mx.eval(v, net.parameters(), opt.state)
        if step % 100 == 0 or step + 1 == s["updates"]:
            losses.append([step + 1, v.item()])
    net.save_weights(str(out / "reader.safetensors"))
    save(
        out / "reader.json",
        {
            "base": REVISION,
            "depth": depth,
            "rank": s["rank"],
            "vocab": vocab,
            "letters": letters,
            "weights": digest(out / "reader.safetensors"),
            "loss": losses,
        },
    )
    return net


def run(name, freeze):
    reg = Path("research/experiments") / (name + ".json")
    s = read(reg)
    if freeze:
        material = science_data.material(s)
        if PANEL.exists() and read(PANEL) != material:
            raise ValueError("Frozen QASC material changed")
        if not PANEL.exists():
            save(PANEL, material)
        return
    idx = read("research/registry.json")
    if idx["datasets"][str(PANEL)]["sha256"] != digest(PANEL) or idx["experiments"][
        name
    ]["registration"] != digest(reg):
        raise ValueError("Unregistered science study")
    if s.get("requires") and not read(Path(s["requires"]) / "decision.json")["passed"]:
        raise ValueError("Prerequisite failed")
    data = read(PANEL)
    b = core.Budget(name, s["limits"])
    out = core.OUT / name
    out.mkdir(parents=True, exist_ok=False)
    result = {"registration": digest(reg), "data": digest(PANEL), "passed": False}
    try:
        model, t, _ = base()
        results = {}
        reports = {}
        rows = data[s["panel"]]
        letters = [core.atom(t, " " + x) for x in science_data.LETTERS]
        prefix = t.encode(
            "After considering the evidence.\n</think>\n\nAnswer:",
            add_special_tokens=False,
        )
        for arm, conf in s["arms"].items():
            depth = conf["depth"]
            if conf.get("fit"):
                net = fit(model, t, data["train"], depth, s, b, out)
            elif depth:
                path = Path(conf["reader"])
                info = read(path / "reader.json")
                if (
                    digest(path / "reader.safetensors") != conf["weights"]
                    or info["base"] != REVISION
                ):
                    raise ValueError("Wrong science reader")
                net = ScienceReader(
                    model.args.hidden_size, info["rank"], info["vocab"], letters
                )
                net.load_weights(str(path / "reader.safetensors"))
            records = []
            for r in rows:
                b.check()
                if (
                    t.encode(
                        science_data.render(t, r["facts"], r["question"], r["options"]),
                        add_special_tokens=False,
                    )
                    != r["prompt_ids"]
                ):
                    raise ValueError("Prompt mismatch")
                start = time.perf_counter()
                view = model
                if depth:
                    a, d = pooled_states(
                        model, t, [*r["facts"], r["question"], *r["options"]], depth
                    )
                    a, d = a.astype(mx.bfloat16), d.astype(mx.bfloat16)
                    n = len(r["facts"])
                    scores, _, _ = net.composition(
                        a[None, :n], d[None, :n], a[None, n], a[None, n + 1 :], depth
                    )
                    view = EvidenceView(model, net, scores)
                v = core.generate(
                    view,
                    t,
                    r["prompt_ids"],
                    s["generation_cap"],
                    b.check,
                    thinking=True,
                )
                total = time.perf_counter() - start
                z = view.project(
                    view.hidden(mx.array([r["prompt_ids"] + prefix]))[:, -1]
                )[0, mx.array(letters)].tolist()
                ans = (
                    science_data.answer(v["visible"], r["options"])
                    if v["complete"]
                    else None
                )
                records.append(
                    dict(
                        **v,
                        id=r["id"],
                        answer=ans,
                        correct=ans == r["answer"],
                        probe=science_data.LETTERS[max(range(8), key=z.__getitem__)],
                        probe_logits=z,
                        end_to_end=total,
                    )
                )
                packed(out / (arm + ".json.gz"), {"records": records})
            results[arm] = records
            reports[arm] = {
                "correct": sum(r["correct"] for r in records),
                "probe": sum(
                    r["probe"] == v["answer"]
                    for r, v in zip(records, rows, strict=True)
                ),
                "seconds": sum(r["end_to_end"] for r in records),
                "complete": sum(r["complete"] for r in records),
            }
        left, right = results.values()
        diff = [
            int(a["correct"]) - int(b["correct"])
            for a, b in zip(left, right, strict=True)
        ]
        a, c = reports.values()
        p = exact_mcnemar(diff.count(1), diff.count(-1))
        result.update(
            status="completed",
            scores=reports,
            wins=diff.count(1),
            losses=diff.count(-1),
            p=p,
            passed=p < 0.05
            and sum(diff) >= s["net_floor"]
            and a["probe"] >= c["probe"] + s["probe_floor"]
            and a["seconds"] <= s["time_ratio"] * c["seconds"],
        )
    except (TimeoutError, InterruptedError, MemoryError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run(args.name, args.freeze)
