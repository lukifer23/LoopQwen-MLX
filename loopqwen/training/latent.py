"""Train the shared categorical state readout on prompts without gold-state tokens."""

import argparse
import gc
import json
import random
import signal
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten
import mlx.nn as nn
import mlx.optimizers as optim

from .. import checkpoint
from ..config import config
from ..data.transitions import ROOT, STATES, build, render
from ..identity import fingerprint, training_identity
from ..io import read, save, digest

OUT = Path("outputs/phase4")


def arm_model(c, arm):
    model, tok, info = checkpoint.setup(c)
    if arm == "probe":
        model.freeze()
    mx.random.seed(c["seed"] + 11)
    model.state_head = nn.Linear(model.args.hidden_size, STATES, bias=False)
    model.checkpointing = True
    return model, tok, info


def logits(model, ids, last, n, arm, extra=0):
    if arm == "recurrent":
        hidden = model.block_states(ids, n + extra)
    else:
        hidden = (model.hidden(ids),)
    return [
        model.state_head(
            mx.fast.rms_norm(
                h[mx.arange(ids.shape[0]), last].astype(mx.float32), None, 1e-6
            )
        )
        for h in hidden
    ]


def data_batch(rows, n, tok, cache, verbal=False):
    tokens = []
    for row in rows:
        key = (row["id"], n, verbal)
        if key not in cache:
            cache[key] = tok.encode(render(row, n, verbal), add_special_tokens=False)
        tokens.append(cache[key])
    width = max(map(len, tokens))
    return (
        mx.array([ids + [0] * (width - len(ids)) for ids in tokens]),
        mx.array([len(ids) - 1 for ids in tokens]),
        mx.array([r["states"][:n] for r in rows]),
    )


def evaluate(model, tok, rows, arm, cache, depths=(1, 2, 3), verbal=False):
    model.eval()
    result = {}
    for n in depths:
        records = []
        begin = perf_counter()
        for i in range(0, len(rows), 8):
            chunk = rows[i : i + 8]
            ids, last, y = data_batch(chunk, n, tok, cache, verbal)
            z = logits(
                model, ids, last, n, arm, extra=int(arm == "recurrent" and n < 6)
            )
            pred = mx.stack([mx.argmax(v, axis=-1) for v in z], axis=1).tolist()
            records.extend(
                dict(id=r["id"], gold=r["states"][:n], pred=p)
                for r, p in zip(chunk, pred, strict=True)
            )
        terminal = n - 1 if arm == "recurrent" else 0
        result[str(n)] = dict(
            seconds=perf_counter() - begin,
            n=len(rows),
            terminal=sum(r["pred"][terminal] == r["gold"][-1] for r in records),
            first_terminal=sum(r["pred"][0] == r["gold"][-1] for r in records),
            per_pass=[
                sum(r["pred"][j] == r["gold"][j] for r in records) for j in range(n)
            ]
            if arm == "recurrent"
            else [],
            excessive=sum(r["pred"][-1] == r["gold"][-1] for r in records)
            if arm == "recurrent" and n < 6
            else None,
            records=records,
        )
    return result


def run(arm, seed, steps=360, anneal=False):
    meta = build()
    c = config(OUT / "configs" / f"p4_{seed}_dense.yaml")
    c.update(mode="direct", passes=1 if arm != "recurrent" else 3, seed=seed)
    model, tok, info = arm_model(c, arm)
    data = {
        s: [
            json.loads(line)
            for line in (ROOT / (s + ".jsonl")).read_text().splitlines()
        ]
        for s in ("train", "development")
    }
    random.Random(seed).shuffle(data["train"])
    cache = {}
    root = Path("checkpoints") / f"p4_state_{seed}_{arm}"
    root.mkdir(exist_ok=True, parents=True)
    ident = fingerprint(
        dict(
            config=training_identity(c, meta),
            experiment=digest(Path(__file__)),
            generator=digest(ROOT / "manifest.json"),
            rendering=digest(
                Path(__file__).resolve().parents[1] / "data/transitions.py"
            ),
            arm=arm,
            install_steps=steps,
        )
    )
    opt = optim.AdamW(learning_rate=0.0003, weight_decay=0.0, bias_correction=True)
    state = dict(
        schema="pointer-state-v1",
        arm=arm,
        manifest=meta,
        trainable=sum(x.size for _, x in tree_flatten(model.trainable_parameters())),
        identity=ident,
        config=c,
        model=info,
        head_parameters=STATES * model.args.hidden_size,
        step=0,
        history=[],
        phase="install",
    )
    if anneal and not (root / "install.json").exists():
        raise ValueError("Complete installation before outcome-only continuation")
    if (root / "latest.json").exists():
        state = checkpoint.restore(root, model, opt)
        if state["identity"] != ident:
            raise ValueError("Latent training identity changed")
    stop = steps + (80 if anneal else 0)

    def loss(m, ids, last, y):
        n = y.shape[1]
        scores = logits(m, ids, last, n, arm)
        pairs = (
            zip(scores, range(n))
            if arm == "recurrent" and not anneal
            else [(scores[-1], n - 1)]
        )
        terms = [
            nn.losses.cross_entropy(z, y[:, j], reduction="mean") for z, j in pairs
        ]
        return sum(terms) / len(terms)

    fn = nn.value_and_grad(model, loss)
    halted = []
    handlers = {
        sig: signal.signal(sig, lambda signum, frame: halted.append(signum))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for step in range(state["step"], stop):
            model.train()
            n = 1 + step % (2 if step < 48 else 3)
            group = [
                data["train"][(8 * step + i) % len(data["train"])] for i in range(8)
            ]
            ids, last, y = data_batch(group, n, tok, cache)
            begin = perf_counter()
            value, g = fn(model, ids, last, y)
            g, norm = optim.clip_grad_norm(g, 1)
            opt.update(model, g)
            mx.eval(model.trainable_parameters(), opt.state, norm, value)
            if not 0 <= value.item() < 100 or not 0 <= norm.item() < 1e8:
                raise FloatingPointError("Invalid state objective")
            state.update(step=step + 1, phase="outcome_only" if anneal else "install")
            state["history"].append(
                dict(
                    loss=value.item(),
                    norm=norm.item(),
                    depth=n,
                    seconds=perf_counter() - begin,
                    shape=list(ids.shape),
                    peak=mx.get_peak_memory(),
                )
            )
            if (step + 1) % 60 == 0 or halted or step + 1 == stop:
                checkpoint.write(root, model, opt, state)
                print(arm, state["step"], value.item(), flush=True)
            if halted:
                raise KeyboardInterrupt("Saved latent checkpoint")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    summary = OUT / f"state-{seed}-{arm}-{'outcome' if anneal else 'install'}.json"
    save(
        root / ("outcome.json" if anneal else "install.json"),
        read(root / "latest.json"),
    )
    suite = evaluate(model, tok, data["development"], arm, cache)
    eligible = (
        arm == "recurrent"
        and all(r["terminal"] / r["n"] > 1 / STATES + 0.15 for r in suite.values())
        and suite["3"]["terminal"] > suite["3"]["first_terminal"]
    )
    transfer = (
        evaluate(model, tok, data["development"], arm, cache, (4, 5, 6))
        if eligible
        else {}
    )
    verbal = (
        evaluate(model, tok, data["development"], arm, cache, verbal=True)
        if eligible
        else {}
    )
    save(
        summary,
        dict(
            identity=ident,
            step=state["step"],
            readout=state["head_parameters"],
            weights=digest(checkpoint.location(root) / "adapters.safetensors"),
            eligible=eligible,
            suite=suite,
            extrapolation=transfer,
            verbal=verbal,
        ),
    )
    del model
    gc.collect()
    mx.clear_cache()
    return eligible


def predict(source, mapping, start, hops):
    if (
        sorted(mapping) != list(range(STATES))
        or start not in range(STATES)
        or hops not in range(1, 7)
    ):
        raise ValueError(
            "Expected an eight-node permutation, valid start, and 1–6 hops"
        )
    gen = checkpoint.location(source)
    state = read(gen / "state.json")
    arm = state["arm"]
    model, tok, _ = arm_model(state["config"], arm)
    checkpoint.restore(gen, model)
    model.eval()
    ids = mx.array(
        [
            tok.encode(
                render({"map": mapping, "start": start}, hops), add_special_tokens=False
            )
        ]
    )
    values = logits(model, ids, mx.array([ids.shape[1] - 1]), hops, arm)
    return dict(
        prompt=tok.decode(ids[0].tolist()),
        predicted_states=[mx.argmax(v).item() for v in values],
        arm=arm,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    p.add_argument("--checkpoint", help="Run a saved categorical readout on one map")
    p.add_argument("--map", type=int, nargs=8)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--hops", type=int, default=3)
    a = p.parse_args()
    if a.checkpoint:
        if a.map is None:
            p.error("--checkpoint needs --map")
        print(json.dumps(predict(a.checkpoint, a.map, a.start, a.hops)))
        raise SystemExit(0)
    for arm in ("probe", "dense", "recurrent"):
        file = OUT / f"state-{a.seed}-{arm}-install.json"
        eligible = read(file)["eligible"] if file.exists() else run(arm, a.seed)
        if eligible and not (OUT / f"state-{a.seed}-{arm}-outcome.json").exists():
            run(arm, a.seed, anneal=True)
