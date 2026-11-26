"""Fit a shared Qwen layer to real question/schema bindings and execute its SQL-tool decisions."""

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

from ..evaluation import table_questions as sql
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..table_parser import TableParser, encode, query_decoder
from ..table_spans import CONDITIONS, OPERATIONS, actions, arguments
from ..table_tool import definition
from ..tool_session import SYSTEM, compact, initial_messages, prompt
from ..tool_session import run as session
from .core import Budget
from .mixed_field import replay_exchange
from .registered import registered
from .tool_fit import restore


def fit(net, model, tok, rows, spec, out, budget):
    at = perf_counter()
    blocks = []
    words = []
    counts = []
    labels = []
    for i, row in enumerate(rows):
        # Training uses the same public table schema. The gold SQL never enters the prompt.
        schemas = [
            {
                "name": "query_table",
                "description": spec["tool_description"],
                "parameters": definition(row["columns"]),
            }
        ]
        text = prompt(
            tok,
            [
                {"role": "system", "content": SYSTEM + compact(schemas)},
                {"role": "user", "content": row["question"] + sql.INSTRUCTION},
            ],
        )
        c, w, nc, nw = encode(model, tok, text, row["question"], row["columns"])
        blocks.append(c)
        words.append(w)
        counts.append([nc, nw])
        target = row["conditions"]
        pads = [[0, 0, 0, 0]] * (CONDITIONS - len(target))
        labels.append(
            [
                row["sql"]["agg"],
                len(target),
                row["sql"]["sel"],
                *[v for cond in target + pads for v in cond],
            ]
        )
        if i % 256 == 255:
            print("schema encodings", i + 1, flush=True)
        budget.check()
    cb = mx.stack(blocks)
    wb = mx.stack(words)
    sizes = mx.array(counts)
    targets = mx.array(labels)
    mx.eval(cb, wb, sizes, targets)
    del blocks, words
    prep = perf_counter() - at

    def loss(m, c, w, n, y):
        agg, count, sel, columns, op, start, end = m(c, w, n[:, 0], n[:, 1])
        ce = nn.losses.cross_entropy
        val = sum(
            ce(z, y[:, i], reduction="mean") for i, z in enumerate((agg, count, sel))
        )
        cs = y[:, 3:].reshape(-1, CONDITIONS, 4)
        mask = mx.arange(CONDITIONS)[None, :] < y[:, 1, None]
        for i, z in enumerate((columns, op, start, end)):
            val += (ce(z, cs[:, :, i]) * mask).sum() / mx.maximum(mask.sum(), 1)
        return val

    rng = random.Random(spec["fit_seed"])
    vg = nn.value_and_grad(net, loss)
    opt = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    logs = []
    norms = []
    for i in range(spec["updates"]):
        ix = mx.array(rng.sample(range(len(rows)), spec["batch"]))
        val, g = vg(net, cb[ix], wb[ix], sizes[ix], targets[ix])
        if i in (0, spec["updates"] - 1):
            norms.append(
                mx.sqrt(
                    sum(
                        mx.square(gr.astype(mx.float32)).sum()
                        for key, gr in tree_flatten(g)
                        if key.startswith("layer.")
                    )
                ).item()
            )
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(net, g)
        mx.eval(val, net.trainable_parameters(), opt.state)
        budget.check()
        if i % 200 == 0 or i + 1 == spec["updates"]:
            logs.append([i + 1, val.item()])
            print("table parser", *logs[-1], flush=True)
    ws = dict(tree_flatten(net.trainable_parameters()))
    path = out / "table-parser.safetensors"
    mx.save_safetensors(str(path), ws)
    save(
        out / "fit.json",
        {
            "training": digest(Path(spec["training"])),
            "weights": digest(path),
            "trainable": sum(x.size for x in ws.values()),
            "prepare_seconds": prep,
            "optimizer_seconds": perf_counter() - at - prep,
            "updates": spec["updates"],
            "depth": spec["depth"],
            "shared_gradient_norms": norms,
            "loss": logs,
        },
    )


def outcome(data, items):
    costs = {}
    for k, arm in data.items():
        by_op = {op: 0 for op in OPERATIONS}
        for r, w in zip(arm, items, strict=True):
            by_op[OPERATIONS[w["sql"]["agg"]]] += r["score"]["success"]
        costs[k] = {
            "success": sum(by_op.values()),
            "by_op": by_op,
            "content": sum(r["score"]["correct_answer"] for r in arm),
            "seconds": sum(r["trajectory"]["seconds"] for r in arm),
            "peak": max(t["peak_bytes"] for r in arm for t in r["trajectory"]["turns"]),
        }
    changes = [
        int(a["score"]["success"]) - int(b["score"]["success"])
        for a, b in zip(data["candidate"], data["control"], strict=True)
    ]
    p = exact_mcnemar(changes.count(1), changes.count(-1))
    a, b = costs["candidate"], costs["control"]
    return {
        "passed": sum(changes) >= 12
        and p <= 0.05
        and min(a["by_op"].values()) >= 3
        and a["content"] > b["content"]
        and a["seconds"] <= b["seconds"]
        and a["peak"] <= b["peak"],
        "gains": changes.count(1),
        "losses": changes.count(-1),
        "p_exact": p,
        "metrics": costs,
    }


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    parent = Path(spec["requires"])
    items = read(spec["material"])["rows"]
    if (
        not read(parent / "decision.json")["passed"]
        or digest(parent / "checksums.json") != spec["prerequisite_sha256"]
    ):
        raise ValueError("Unqualified donor")
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                model, tok = restore(parent)
                mx.random.seed(spec["fit_seed"])
                net = TableParser(model, spec["rank"], spec["depth"])
                if (out / "fit.json").exists():
                    f = read(out / "fit.json")
                    path = out / "table-parser.safetensors"
                    if (
                        digest(path) != f["weights"]
                        or digest(Path(spec["training"])) != f["training"]
                        or f["updates"] != spec["updates"]
                    ):
                        raise ValueError("Changed fitted parser")
                    net.load_weights(str(path), strict=False)
                else:
                    fit(
                        net,
                        model,
                        tok,
                        read(spec["training"])["rows"],
                        spec,
                        out,
                        budget,
                    )
                net.eval()
            else:
                model, tok, _ = base()
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            rs = []
            for i, w in enumerate(items):
                tools = sql.tools_for(w)
                if tools[0].description != spec["tool_description"]:
                    raise ValueError("Training and serving descriptions differ")
                evidence = {}
                tr = session(
                    model,
                    tok,
                    w["question"] + sql.INSTRUCTION,
                    tools,
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    first_turn_decoder=query_decoder(
                        net, w["question"], tools[0], evidence
                    )
                    if side == "candidate"
                    else None,
                    progress=lambda _: budget.check(),
                )
                rs.append(
                    {
                        "id": w["id"],
                        "trajectory": tr,
                        "score": sql.score(w, tr),
                        "plan": evidence,
                    }
                )
                if i % 8 == 7:
                    print(
                        side, i + 1, sum(r["score"]["success"] for r in rs), flush=True
                    )
            data[side] = rs
            packed(out / (side + ".json.gz"), rs)
            if side == "candidate":
                del net
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(out / "decision.json", outcome(data, items))
    finally:
        budget.close()


def verify(root, d):
    spec = read(root / "registration.json")
    items = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    f = read(root / "fit.json")
    train = read(spec["training"])["rows"]
    assert (
        f["weights"] == digest(root / "table-parser.safetensors")
        and f["training"] == digest(Path(spec["training"]))
        and f["updates"] == spec["updates"]
        and f["depth"] == spec["depth"]
        and all(x > 0 for x in f["shared_gradient_norms"])
    )
    assert {r["table_id"] for r in train}.isdisjoint(w["table_id"] for w in items)
    assert all(r["conditions"] == actions(r) for r in train)
    data = {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for r, w in zip(data[side], items, strict=True):
            tools = sql.tools_for(w)
            tr = r["trajectory"]
            replay_exchange(
                tok,
                tools,
                initial_messages(w["question"] + sql.INSTRUCTION, tools),
                tr,
                spec,
            )
            assert r["id"] == w["id"] and r["score"] == sql.score(w, tr)
            if side == "candidate":
                p = r["plan"]
                args = arguments(
                    w["question"],
                    w["columns"],
                    p["select"],
                    p["aggregate"],
                    p["conditions"],
                )
                assert tr["turns"][0]["call"] == {
                    "name": "query_table",
                    "arguments": args,
                }
    assert outcome(data, items) == d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
