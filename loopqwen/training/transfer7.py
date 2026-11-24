"""Bounded full-vocabulary next-digit adaptation of the qualified recurrent checkpoint."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import math
import signal
import subprocess

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import psutil
from .. import checkpoint
from ..data import pointer5, pointer6
from ..data.tokenize import physical_batches
from ..identity import fingerprint
from ..io import read, save, digest
from . import pointer5 as kernel
from .exposure import ARM, S, OUT, verify_optimizer
from .mechanism import generation, model_for, provenance

T = S["transfer"]
P = read("experiments/phase7_transfer_parent.json")
ARM_T = replace(
    ARM,
    name="verbal",
    objective="state_and_full_vocab",
    milestones=tuple(T["milestones"]),
)


def prompt(row, style):
    phrases = {
        "room": (
            "Room links",
            "room {} leads to room {}",
            "Starting room",
            "Destination room",
        ),
        "document": (
            "Document references",
            "document {} cites document {}",
            "Initial document",
            "Current document",
        ),
        "pointer": (
            "Array pointers",
            "entry {} points to entry {}",
            "Initial index",
            "Current index",
        ),
        "state machine": (
            "Machine transitions",
            "state {} moves to state {}",
            "Initial state",
            "Current state",
        ),
    }
    title, edge, start, cue = phrases[style]
    edges = "; ".join(edge.format(i, row["map"][i]) for i in row["order"])
    return f"{title}: {edges}. {start}: {row['start']}.\n{cue}: "


def example(row, style, tok, digits):
    text = prompt(row, style)
    ids = tok.encode(text, add_special_tokens=False)
    for state in set(row["states"][:3]):
        if ids + [digits[state]] != tok.encode(
            text + str(state), add_special_tokens=False
        ):
            raise ValueError("Next-digit boundary changed")
    return dict(ids=ids, row=row, style=style)


def digit_ids(tok):
    ids = [tok.encode(str(i), add_special_tokens=False) for i in range(8)]
    if any(len(v) != 1 for v in ids):
        raise ValueError("Digit must occupy one vocabulary token")
    return [a[0] for a in ids]


def vocab(m, h, last):
    z = m.project(m.suffix(h)[mx.arange(last.size), last])
    if z.shape[-1] != 248320:
        raise ValueError("Expected complete pinned vocabulary")
    return z


def updater(m, optimizer, digits):
    def loss(model, x, last, labels):
        n = labels.shape[1]
        h = model.prefix_hidden(x)
        for _ in range(n):
            h = model.shared_block(h)
        state = labels[:, -1]
        ce = nn.losses.cross_entropy(kernel.readout(model, h, last), state).sum()
        lm = mx.fast.cross_entropy(vocab(model, h, last), mx.array(digits)[state]).sum()
        return T["state_weight"] * ce + T["vocab_weight"] * lm, mx.stack([ce, lm])

    vg = nn.value_and_grad(m, loss)

    def update(group, depth):
        start = perf_counter()
        grads = total = None
        positions = 0
        for physical in physical_batches(group, T["cap_positions"], next_token=False):
            x, last = kernel.inputs(physical)
            y = mx.array([q["row"]["states"][:depth] for q in physical])
            (_, ce), g = vg(m, x, last, y)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            total = ce if total is None else total + ce
            positions += x.size
            mx.eval(grads, total)
        grads, norm = optim.clip_grad_norm(
            tree_map(lambda a: a / len(group), grads), S["clip"]
        )
        mx.eval(norm, total)
        if not math.isfinite(norm.item()) or not mx.all(mx.isfinite(total)).item():
            raise FloatingPointError("Nonfinite transfer update")
        optimizer.update(m, grads)
        mx.eval(m.trainable_parameters(), optimizer.state)
        return dict(
            seconds=perf_counter() - start,
            depth=depth,
            maps=len(group),
            ce=(total / len(group)).tolist(),
            gradient=norm.item(),
            positions=positions,
            site_positions=positions * (18 + 6 * depth),
            input_tokens=sum(len(r["ids"]) for r in group),
            ids=[r["row"]["example_id"] for r in group],
            token_digest=fingerprint([r["ids"] for r in group]),
            peak=mx.get_peak_memory(),
            available=psutil.virtual_memory().available,
        )

    return update


def unsafe():
    return psutil.virtual_memory().available < S[
        "available_gib"
    ] * 2**30 or subprocess.check_output(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
    ).strip() in ("2", "4")


def run():
    from ..evaluation.transfer7 import assess

    dec = read(OUT / "decision.json")
    if not dec["robust"] or dec["step"] != P["parent_absolute_step"]:
        raise ValueError("Registered mechanism prerequisite failed")
    parent = generation(ARM, dec["step"])
    if digest(parent / "adapters.safetensors") != P["parent_weights"]:
        raise ValueError("Unexpected transfer parent")
    m, t, c = model_for(ARM_T)
    opt = optim.AdamW(learning_rate=T["lr"], weight_decay=0, bias_correction=True)
    prov = provenance(c, ARM_T)
    identity = fingerprint([T, P, prov, digest(Path(__file__))])
    ARM_T.root.mkdir(parents=True, exist_ok=True)
    if (ARM_T.root / "latest.json").exists():
        st = checkpoint.restore(ARM_T.root, m, opt)
        if st["identity"] != identity:
            raise ValueError("Incompatible transfer resume")
        if st["step"]:
            verify_optimizer(m, opt, st["step"], T["lr"])
    else:
        checkpoint.restore(parent, m)
        st = dict(
            identity=identity,
            provenance=prov,
            step=0,
            history=[],
            config=c,
            parent=P,
            policy=T,
            status="running",
            wall_seconds=0,
        )
    if st["status"] in ("completed", "unsafe", "time_limit"):
        del m, t, opt
        return assess(st)
    digits = digit_ids(t)
    fn = updater(m, opt, digits)
    schedule = pointer6.epoch_batches(pointer5.rows("train"), T["seed"], 0)
    interrupted = []
    handlers = {
        s: signal.signal(s, lambda n, f: interrupted.append(n))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    start = perf_counter()
    elapsed = sum(r["seconds"] for r in st["history"])
    mx.reset_peak_memory()
    m.train()

    def mark():
        nonlocal start
        st["wall_seconds"] += perf_counter() - start
        checkpoint.write(ARM_T.root, m, opt, st)
        path = ARM_T.root / f"at-{st['step']}.json"
        if not path.exists():
            save(path, read(ARM_T.root / "latest.json"))
        start = perf_counter()

    try:
        for i in range(st["step"], T["updates"]):
            if unsafe():
                st["status"] = "unsafe"
                break
            if (
                elapsed + max([3.0] + [2 * r["seconds"] for r in st["history"][-12:]])
                >= T["max_update_seconds"]
            ):
                st["status"] = "time_limit"
                break
            group = [
                example(
                    pointer6.ordered(r, "randomized", T["seed"], 7),
                    T["training_renderings"][(i * T["batch"] + j) % 3],
                    t,
                    digits,
                )
                for j, r in enumerate(schedule[i])
            ]
            r = fn(group, 1 + i % 3)
            st["history"].append(r)
            st["step"] = i + 1
            elapsed += r["seconds"]
            if int(opt.step.item()) != st["step"]:
                raise ValueError("Optimizer cursor mismatch")
            if r["peak"] > T["max_peak_gib"] * 2**30 or unsafe():
                st["status"] = "unsafe"
                break
            if interrupted:
                st["status"] = "interrupted"
                break
            if st["step"] in T["milestones"][:-1]:
                mark()
        else:
            st["status"] = "completed"
    except FloatingPointError:
        st["status"] = "unsafe"
        raise
    finally:
        mark()
        for sig, h in handlers.items():
            signal.signal(sig, h)
    if interrupted:
        raise KeyboardInterrupt
    print(
        "Transfer", st["status"], st["step"], "updates", elapsed, "seconds", flush=True
    )
    del fn, m, t, opt
    mx.clear_cache()
    return assess(st)
