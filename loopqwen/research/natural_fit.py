"""Train passage relevance on annotated spans and calibrate native reader choice."""

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from ..attention_memory import PassageScorer
from ..context_policy import features
from ..io import read, save
from . import memory_data


def fit(s, model, t, rows, docs, budget, root):
    mx.random.seed(s["seed"])
    embed = model.text.model.embed_tokens
    scorer = PassageScorer(model.args.hidden_size, **s["architecture"])
    opt = optim.Adam(s["reader_rate"])

    def objective(net, x, lo, hi):
        logits = 30 * net(x)
        starts = mx.arange(logits.size) * net.stride
        centers = starts + net.span_width // 2
        positive = mx.abs(centers - (lo + hi) // 2) <= net.stride
        return mx.logsumexp(logits) - mx.logsumexp(
            mx.where(positive, logits, -float("inf"))
        )

    vg = nn.value_and_grad(scorer, objective)
    losses = []
    trained = []
    for r in rows:
        budget.check()
        lo, hi = r["answer_start"], r["answer_stop"]
        if (
            lo < s["architecture"]["span_width"] // 2
            or hi
            > len(r["ids"])
            - s["architecture"]["query_length"]
            - s["architecture"]["span_width"] // 2
        ):
            continue
        x = mx.stop_gradient(embed(mx.array(r["ids"])))
        v, g = vg(scorer, x, lo, hi)
        g, _norm = optim.clip_grad_norm(g, 1.0)
        opt.update(scorer, g)
        mx.eval(scorer.parameters(), opt.state, v)
        if not mx.isfinite(v).item():
            raise FloatingPointError("Nonfinite reader training objective")
        losses.append(v.item())
        trained.append(r["id"])
    if len(trained) < 0.75 * len(rows):
        raise ValueError("Insufficient localized examples")
    scorer.save_weights(str(root / "prose-scorer.safetensors"))
    n = s["routing_examples"]
    f = []
    math = read(s["routing_math"])
    for i in range(n):
        budget.check()
        reg = memory_data.make_item(
            t,
            s["seed"] + 20000 + i,
            512 + (i % 3) * 256,
            0.2 + 0.25 * (i % 3),
            ("table", "statement", "note")[i % 3],
        )
        doc = docs[i]
        art = rows[i]["ids"] if i % 2 else doc["ids"][: doc["prefix"]]
        f.extend(
            (
                features(embed, math[i]["prompt_ids"]),
                features(embed, reg["prompt_ids"]),
                features(embed, art),
            )
        )
    x = mx.stack(f)
    y = mx.tile(mx.array([0, 1, 2]), n)
    cl = nn.Linear(x.shape[1], 3)
    opt = optim.Adam(0.003)
    vg = nn.value_and_grad(cl, lambda net: nn.losses.cross_entropy(net(x), y).mean())
    for _ in range(48):
        budget.check()
        v, g = vg(cl)
        opt.update(cl, g)
        mx.eval(cl.parameters(), opt.state)
    cl.save_weights(str(root / "selector.safetensors"))
    save(
        root / "reader-training.json",
        {
            "questions": trained,
            "loss": losses,
            "route_examples": 3 * n,
            "route_updates": 48,
        },
    )
