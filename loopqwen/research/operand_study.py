"""Train a recurrent operand and operator planner from verified programs on native MLX."""

import argparse
import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ..arithmetic_actions import OPS, as_steps, mentions
from ..calculator_tool import execute
from ..evaluation import program_questions
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..operand_planner import OperandPlanner, encode_document, padded, rollout, slot
from ..structured_call import decoder
from ..tool_session import initial_messages, prompt
from ..tool_session import run as session
from .core import Budget
from .registered import registered
from .tool_fit import restore


def features(model, tok, rows, budget):
    docs = []
    for i, r in enumerate(rows):
        request = r["question"] + program_questions.INSTRUCTION
        text = prompt(tok, initial_messages(request, program_questions.tools_for(r)))
        docs.append(encode_document(model, tok, text, r["question"]))
        budget.check()
        if i % 256 == 255:
            print("training encodings", i + 1, flush=True)
    n = max(16, max(len(v) for _, v in docs))
    hv = [padded(h, v, n) for h, v in docs]
    h = mx.stack([x[0] for x in hv])
    v = mx.stack([x[1] for x in hv])
    mx.eval(h, v)
    return h, v, [len(values) for _, values in docs], n


def fit(spec, planner, model, tok, rows, out, budget):
    at = perf_counter()
    bank, values, lengths, n = features(model, tok, rows, budget)
    encoded = perf_counter() - at
    acts = [
        [[OPS.index(o), slot(l, n), slot(r, n)] for o, l, r in w["actions"]]
        for w in rows
    ]
    sources = [
        [i for i, r in enumerate(rows) if r["source"] == key]
        for key in ("GSM8K", "MAWPS")
    ]

    def loss(net, h, v, counts, hist, steps, targets, stop):
        logits, halt = net(h, v, counts, hist, steps)
        m = mx.arange(h.shape[0])
        op, l, r = targets.T
        direct = logits[m, op, l, r]
        reverse = logits[m, op, r, l]
        correct = mx.where(
            ((op == 0) | (op == 2)) & (l != r), mx.logaddexp(direct, reverse), direct
        )
        cross = mx.logsumexp(logits.reshape(h.shape[0], -1), axis=-1) - correct
        return cross.mean() + nn.losses.binary_cross_entropy(
            halt, stop, with_logits=True, reduction="mean"
        )

    rng = random.Random(spec["fit_seed"])
    vg = nn.value_and_grad(planner, loss)
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    log = []
    shared_gradients = []
    for step in range(spec["updates"]):
        batch = [
            j for source in sources for j in rng.sample(source, spec["batch"] // 2)
        ]
        k = [rng.randrange(len(acts[i])) for i in batch]
        depth = max(k)
        hist = mx.array(
            [
                acts[i][:s] + [[0, 0, 0]] * (depth - s)
                for i, s in zip(batch, k, strict=True)
            ],
            dtype=mx.int32,
        ).reshape(len(k), depth, 3)
        target = mx.array([acts[i][s] for i, s in zip(batch, k, strict=True)])
        stop = mx.array(
            [float(s + 1 == len(acts[i])) for i, s in zip(batch, k, strict=True)]
        )
        ix = mx.array(batch)
        loss_value, grad = vg(
            planner,
            bank[ix],
            values[ix],
            mx.array([lengths[i] for i in batch]),
            hist,
            mx.array(k),
            target,
            stop,
        )
        if step in (0, spec["updates"] - 1):
            shared_norm = mx.sqrt(
                sum(
                    mx.square(v.astype(mx.float32)).sum()
                    for k, v in tree_flatten(grad)
                    if k.startswith("layer.")
                )
            )
            shared_gradients.append([step + 1, shared_norm.item()])
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        opt.update(planner, grad)
        mx.eval(loss_value, planner.trainable_parameters(), opt.state)
        budget.check()
        if step % 100 == 0 or step + 1 == spec["updates"]:
            log.append([step + 1, loss_value.item()])
            print("planner", *log[-1], flush=True)
    weights = dict(tree_flatten(planner.trainable_parameters()))
    file = out / "planner.safetensors"
    mx.save_safetensors(str(file), weights)
    save(
        out / "fit.json",
        dict(
            training=digest(Path(spec["training"])),
            weights=digest(file),
            trainable=sum(v.size for v in weights.values()),
            updates=spec["updates"],
            native_layer=15,
            shared_depth=planner.depth,
            shared_gradient_norms=shared_gradients,
            prepare_s=encoded,
            optimizer_s=perf_counter() - at - encoded,
            loss=log,
        ),
    )


def action_decoder(planner, question, record):
    def action(model, tok, text):
        h, values = encode_document(model, tok, text, question)
        steps, actions = rollout(planner, h, values)
        record.update(values=values, actions=actions, steps=steps)
        return {"name": "calculate", "arguments": {"steps": steps}}

    return decoder(action)


def run(name):
    from scripts.tool_study import summarize

    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    parent = Path(spec["requires"])
    if (
        digest(parent / "checksums.json") != spec["prerequisite_sha256"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Unqualified language readout")
    out.mkdir(exist_ok=True, parents=True)
    rows = read(spec["training"])["rows"]
    panel = read(spec["material"])["rows"]
    data = {}
    budget = Budget(name, spec["limits"])
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                model, tok = restore(parent)
                mx.random.seed(spec["fit_seed"])
                planner = OperandPlanner(model, spec["rank"], spec["depth"])
                weights = out / "planner.safetensors"
                if (out / "fit.json").exists():
                    f = read(out / "fit.json")
                    if digest(weights) != f["weights"] or f["training"] != digest(
                        Path(spec["training"])
                    ):
                        raise ValueError("Changed saved planner")
                    planner.load_weights(str(weights), strict=False)
                else:
                    fit(spec, planner, model, tok, rows, out, budget)
                planner.eval()
            else:
                model, tok, _ = base()
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            rs = []
            for i, w in enumerate(panel):
                evidence = {}
                request = w["question"] + program_questions.INSTRUCTION
                tr = session(
                    model,
                    tok,
                    request,
                    program_questions.tools_for(w),
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    first_turn_decoder=action_decoder(planner, w["question"], evidence)
                    if side == "candidate"
                    else None,
                    progress=lambda _: budget.check(),
                )
                rs.append(
                    dict(
                        id=w["id"],
                        trajectory=tr,
                        score=program_questions.score(w, tr),
                        plan=evidence,
                    )
                )
                if i % 8 == 7:
                    print(
                        side,
                        i + 1,
                        sum(r["score"]["correct_answer"] for r in rs),
                        flush=True,
                    )
            packed(out / (side + ".json.gz"), rs)
            data[side] = rs
            if side == "candidate":
                del planner
            del model, tok
            gc.collect()
            mx.clear_cache()
        d = summarize(data, spec)
        save(out / "decision.json", d)
        print(d, flush=True)
    finally:
        budget.close()


def verify(root, d):
    from fractions import Fraction

    from scripts.research_archive import read as archive_read
    from scripts.research_archive import require, sha
    from scripts.tool_audit import verify as audit

    spec = read(root / "registration.json")
    audit(root, d, archive_read, sha, require)
    f = read(root / "fit.json")
    assert (
        f["weights"] == digest(root / "planner.safetensors")
        and f["training"] == digest(Path(spec["training"]))
        and f["updates"] == spec["updates"]
    )
    for w in read(spec["training"])["rows"]:
        ns = [v for _, _, v in mentions(w["question"])]
        assert execute(as_steps(ns, w["actions"]))[-1] == Fraction(w["gold"])
    for r, w in zip(
        unpack(root / "candidate.json.gz"), read(spec["material"])["rows"], strict=True
    ):
        p = r["plan"]
        ns = [v for _, _, v in mentions(w["question"])]
        assert (
            p["values"] == ns
            and as_steps(ns, p["actions"]) == p["steps"]
            and 1 <= len(p["steps"]) <= 6
        )
        assert r["trajectory"]["turns"][0]["call"] == dict(
            name="calculate", arguments=dict(steps=p["steps"])
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
