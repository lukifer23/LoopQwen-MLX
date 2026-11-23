"""Categorical optimization through the real shared Qwen block, with one update per logical batch."""

import gc
import json
import math
import random
import signal
import subprocess
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import psutil

from .. import checkpoint
from ..config import config
from ..data.pointer5 import REG, ROOT, build, prompt, rows
from ..data.tokenize import physical_batches
from ..identity import dependencies, fingerprint, training_identity
from ..io import digest, read, save
from .latent import arm_model

OUT = Path("outputs/phase5")
REC = ("recurrent", "terminal", "bridge", "outcome")
SPEC = read(REG)


def architecture(seed, arm, policy):
    c = config("configs/pilot_loop.yaml")
    c.update(
        seed=seed,
        passes=3 if arm in REC else 1,
        mode="direct",
        gradient_checkpointing=policy != "off",
        checkpoint_policy=policy if policy == "block" else "all",
        rank=SPEC["rank"],
        lora_alpha=SPEC["lora_alpha"],
        start=SPEC["start"],
        length=SPEC["length"],
    )
    if c["base_revision"] != SPEC["base_revision"]:
        raise ValueError("Base revision changed")
    model, tok, info = arm_model(c, arm)
    model.checkpointing = c["gradient_checkpointing"]
    if arm == "bridge":
        model.bridge_in = nn.Linear(2048, SPEC["bridge_rank"], bias=False)
        model.bridge_out = nn.Linear(SPEC["bridge_rank"], 1024, bias=False)
        model.bridge_out.weight = mx.zeros_like(model.bridge_out.weight)
    if arm == "multi":
        model.depth_scale = mx.ones((3, 1024))
    return model, tok, c, info


def normalize(h):
    return mx.fast.rms_norm(h.astype(mx.float32), None, 1e-6)


def reenter(m, h, p, arm):
    if arm != "bridge":
        return h
    delta = m.bridge_out(
        m.bridge_in(mx.concatenate([normalize(h), normalize(p)], axis=-1))
    )
    return (h + delta).astype(h.dtype)


def readout(m, h, last):
    return m.state_head(normalize(h[mx.arange(h.shape[0]), last]))


def outputs(m, ids, last, n, arm):
    if arm == "full":
        return [readout(m, m.hidden(ids), last)]
    if arm == "multi":
        h = next(m.block_states(ids, 1))
        v = normalize(h[mx.arange(h.shape[0]), last])
        return [m.state_head(v * m.depth_scale[j]) for j in range(n)]
    if arm == "bridge":
        p = m.prefix_hidden(ids)
        h = p
        result = []
        for j in range(n):
            h = m.shared_block(reenter(m, h, p, arm) if j else h)
            result.append(readout(m, h, last))
        return result
    return [readout(m, h, last) for h in m.block_states(ids, n if arm in REC else 1)]


def request_rows(group, depth, arm, tok, cache):
    out = []
    for r in group:
        text = prompt(r, None if arm in REC or arm == "multi" else depth)
        if text not in cache:
            cache[text] = tok.encode(text, add_special_tokens=False)
        out.append({"ids": cache[text], "row": r})
    return out


def inputs(group):
    w = max(16, max(len(r["ids"]) for r in group))
    return mx.array([r["ids"] + [0] * (w - len(r["ids"])) for r in group]), mx.array(
        [len(r["ids"]) - 1 for r in group]
    )


def schedule(seed, step, train):
    epoch, index = divmod(step * SPEC["batch"], len(train))
    order = list(range(len(train)))
    random.Random(seed + epoch).shuffle(order)
    n = 1 + (step + epoch) % (2 if step < SPEC["early_updates"] else 3)
    return n, [train[i] for i in order[index : index + SPEC["batch"]]]


def sites(arm, n):
    return 24 if arm == "full" else 10 + 6 * (n if arm in REC else 1)


def norm2(leaves):
    return sum(mx.sum(v.astype(mx.float32).square()) for v in leaves)


def make_update(m, opt, arm):
    def objective(model, x, last, y):
        n = y.shape[1]
        logits = outputs(model, x, last, n, arm)
        if arm in ("recurrent", "bridge", "multi"):
            scored = list(zip(logits, range(n), strict=True))
        else:
            scored = [(logits[-1], n - 1)]
        ce = mx.stack([nn.losses.cross_entropy(z, y[:, j]).sum() for z, j in scored])
        acc = mx.stack([(mx.argmax(z, axis=-1) == y[:, j]).sum() for z, j in scored])
        return ce.mean(), ce, acc

    vg = nn.value_and_grad(m, objective)

    def update(group, n, cap):
        begin = perf_counter()
        grads = cs = ac = None
        positions = 0
        shapes = []
        for physical in physical_batches(group, cap, next_token=False):
            x, last = inputs(physical)
            y = mx.array([r["row"]["states"][:n] for r in physical])
            (_, ce, a), g = vg(m, x, last, y)
            grads = g if grads is None else tree_map(lambda u, v: u + v, grads, g)
            cs = ce if cs is None else cs + ce
            ac = a if ac is None else ac + a
            positions += x.size
            shapes.append(list(x.shape))
            mx.eval(grads, cs, ac)
        size = len(group)
        grads = tree_map(lambda x: x / size, grads)
        flat = dict(tree_flatten(grads))
        old = dict(tree_flatten(m.trainable_parameters()))
        head_norm = mx.sqrt(norm2([flat["state_head.weight"]]))
        adapter_keys = [k for k in flat if k.endswith(("lora_a", "lora_b"))]
        adapter_grad = (
            mx.sqrt(norm2([flat[k] for k in adapter_keys]))
            if adapter_keys
            else mx.array(0.0)
        )
        bridge_keys = [k for k in flat if k.startswith("bridge_")]
        bridge_grad = (
            mx.sqrt(norm2([flat[k] for k in bridge_keys]))
            if bridge_keys
            else mx.array(0.0)
        )
        grads, norm = optim.clip_grad_norm(grads, SPEC["clip"])
        mx.eval(norm, cs, ac, head_norm, adapter_grad, bridge_grad)
        if not math.isfinite(norm.item()) or not mx.all(mx.isfinite(cs)).item():
            raise FloatingPointError("Invalid categorical loss or gradient")
        opt.update(m, grads)
        new = dict(tree_flatten(m.trainable_parameters()))
        delta = (
            mx.sqrt(norm2([new[k] - old[k] for k in adapter_keys]))
            if adapter_keys
            else mx.array(0.0)
        )
        mx.eval(m.trainable_parameters(), opt.state, delta)
        return dict(
            depth=n,
            maps=size,
            labels=size * cs.size,
            intermediate=size * (cs.size - 1),
            ce=(cs / size).tolist(),
            correct=ac.tolist(),
            gradient=norm.item(),
            head_gradient=head_norm.item(),
            adapter_gradient=adapter_grad.item(),
            bridge_gradient=bridge_grad.item(),
            adapter_update=delta.item(),
            seconds=perf_counter() - begin,
            positions=positions,
            site_positions=positions * sites(arm, n),
            shapes=shapes,
            peak=mx.get_peak_memory(),
            active=mx.get_active_memory(),
            cached=mx.get_cache_memory(),
            available=psutil.virtual_memory().available,
            rss=psutil.Process().memory_info().rss,
        )

    return update


def adam():
    return optim.AdamW(learning_rate=SPEC["lr"], weight_decay=0, bias_correction=True)


def safe():
    if psutil.virtual_memory().available < SPEC["min_available_gib"] * 2**30:
        mx.clear_cache()
        if psutil.virtual_memory().available < SPEC["min_available_gib"] * 2**30:
            raise MemoryError("Insufficient available unified memory")


def profile(seed):
    build()
    train = rows("train")
    stats = []
    for policy in ("off", "block", "all"):
        m, tok, _, _ = architecture(seed, "recurrent", policy)
        m.train()
        update = make_update(m, adam(), "recurrent")
        cache = {}
        for size in (4, 8, 16):
            for n in (1, 2, 3):
                safe()
                mx.reset_peak_memory()
                groups = request_rows(train[:size], n, "recurrent", tok, cache)
                updates = [update(groups, n, SPEC["max_positions"]) for _ in range(3)]
                stats.append(
                    dict(
                        policy=policy,
                        batch=size,
                        depth=n,
                        warmup=updates[0],
                        measured=updates[1:],
                    )
                )
        del update, m
        gc.collect()
        mx.clear_cache()
    eligible = {}
    for policy in ("off", "block", "all"):
        rec = [r for r in stats if r["policy"] == policy and r["batch"] == 8]
        if all(
            v["peak"] < SPEC["max_peak_gib"] * 2**30
            and v["available"] > SPEC["min_available_gib"] * 2**30
            for r in rec
            for v in r["measured"]
        ):
            eligible[policy] = sum(
                median(x["seconds"] for x in r["measured"]) for r in rec
            )
    if not eligible:
        raise MemoryError("No viable batch-eight categorical policy")
    save(
        OUT / "profile.json",
        dict(
            seed=seed,
            selected=min(eligible, key=eligible.get),
            times=eligible,
            profiles=stats,
        ),
    )


def root_for(seed, arm):
    return Path("checkpoints") / f"p5_{seed}_{arm}"


def identity(c, arm):
    import importlib.util

    files = [
        Path(importlib.util.find_spec(n).origin)
        for n in (
            "loopqwen.training.latent",
            "loopqwen.training.pointer5",
            "loopqwen.data.pointer5",
            "loopqwen.data.transitions",
        )
    ]
    return fingerprint(
        dict(
            base=training_identity(c, read(ROOT / "manifest.json")),
            registration=SPEC,
            arm=arm,
            head="rms32-last-1024x8",
            sources={p.name: digest(p) for p in files},
        )
    )


def install(seed, arm, until):
    from ..evaluation.pointer5 import milestone

    build()
    train = rows("train")
    pol = read(OUT / "profile.json")["selected"]
    m, tok, c, info = architecture(seed, arm, pol)
    opt = adam()
    root = root_for(seed, arm)
    root.mkdir(exist_ok=True, parents=True)
    ident = identity(c, arm)
    state = dict(
        schema=5,
        identity=ident,
        arm=arm,
        seed=seed,
        config=c,
        base=info["revision"],
        capacity=sum(x.size for _, x in tree_flatten(m.trainable_parameters())),
        registration=SPEC,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dependencies=dependencies(),
        step=0,
        history=[],
        status="started",
    )
    if (root / "latest.json").exists():
        state = checkpoint.restore(root, m, opt)
        if state["identity"] != ident:
            raise ValueError(
                "Incompatible pointer checkpoint; use a distinct experiment"
            )
    elif arm == "outcome":
        parent = root_for(seed, "recurrent")
        gen = parent / read(parent / "at-480.json")["directory"]
        original = checkpoint.restore(gen, m, opt)
        state.update(
            step=480,
            parent=digest(gen / "adapters.safetensors"),
            parent_optimizer=digest(gen / "optimizer.safetensors"),
        )
        if original["step"] != 480:
            raise ValueError("Outcome lineage requires completed installation")
    marks = [560] if arm == "outcome" else SPEC["milestones"]
    for step in marks:
        if step <= min(state["step"], until):
            milestone(seed, arm, step)
    if state["step"] >= until:
        return
    cache = {}
    fn = make_update(m, opt, arm)
    stopped = []
    handlers = {
        s: signal.signal(s, lambda sig, frame: stopped.append(sig))
        for s in (signal.SIGTERM, signal.SIGINT)
    }
    mx.reset_peak_memory()
    try:
        for step in range(state["step"], until):
            safe()
            m.train()
            n, group = schedule(seed, step, train)
            r = fn(request_rows(group, n, arm, tok, cache), n, SPEC["max_positions"])
            r.update(ids=[x["example_id"] for x in group], step=step + 1)
            state["history"].append(r)
            state.update(step=step + 1, status="running")
            if step + 1 in marks or (step + 1) % 60 == 0 or stopped:
                state["status"] = (
                    "completed"
                    if step + 1 == marks[-1]
                    else "interrupted"
                    if stopped
                    else "running"
                )
                checkpoint.write(root, m, opt, state)
                if step + 1 in marks:
                    marker = root / f"at-{step + 1}.json"
                    if marker.exists():
                        raise FileExistsError(marker)
                    save(marker, read(root / "latest.json"))
                    milestone(seed, arm, step + 1, m, tok)
                print(arm, step + 1, r["ce"], flush=True)
            if stopped:
                raise KeyboardInterrupt("Saved categorical optimizer and cursor")
    except MemoryError:
        checkpoint.write(root, m, opt, state)
        raise
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
