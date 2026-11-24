"""Bounded verified-trajectory collection from the unrestricted Phase 7 vocabulary model."""

import math
import random
import re
import signal
import subprocess
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import psutil
from .. import checkpoint
from ..data import pointer5, pointer6, transitions
from ..data.curriculum import pools, historical
from ..data.tokenize import batch
from ..evaluation.answers import numeric
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import read, save, digest, checkpoint_location
from .exposure import ARM
from .losses import loss_sum
from .mechanism import assets, model_for
from .transfer7 import prompt

REG = Path("experiments/rsi_verified_trace.json")
ROOT = Path("outputs/rsi_trace1")


def verdict(mapping, start, text, stop):
    gold = transitions.states(mapping, start, 3)
    match = re.fullmatch(
        r"\s*([0-7])\s*(?:,|->|→)\s*([0-7])\s*(?:,|->|→)\s*([0-7])[.!]?\s*", text
    )
    final = match[3] if match else numeric(text)
    if stop == "cap":
        status = "truncated"
    elif match:
        status = (
            "verified"
            if list(map(int, match.groups())) == gold
            else "incorrect_transition"
        )
    elif final is not None:
        status = "missing_steps"
    elif not text.strip():
        status = "unparseable"
    else:
        status = "unsupported"
    return dict(
        status=status,
        final=final,
        final_correct=final == str(gold[-1]),
        verified=status == "verified",
    )


def material(seed):
    prior, _ = pools()
    excluded = historical() | {r["map_id"] for v in prior.values() for r in v}
    rng = random.Random(seed)
    rows = []
    while len(rows) < 32:
        r = pointer5.instance(rng.sample(range(8), 8), rng.randrange(8))
        if r["map_id"] in excluded or pointer5.cycle(r) < 4:
            continue
        excluded.add(r["map_id"])
        rows.append(pointer6.ordered(r, "random0", seed))
    return rows[:16], rows[16:]


def request(r):
    return (
        prompt(r, "room").rsplit("\n", 1)[0]
        + " Follow exactly three links. Write the three visited room numbers in order, separated by commas.\nVisited rooms:"
    )


def sample(m, t, r, candidate, temperature, policy, guard):
    ids = t.encode(request(r), add_special_tokens=False)
    eos = set(t.eos_token_ids) | set(t.encode("<|im_end|>", add_special_tokens=False))
    seed = int(fingerprint([policy["seed"], r["map_id"], candidate])[:8], 16)
    mx.random.seed(seed)
    m.eval()
    cache = m.make_cache()
    x = mx.array([ids])
    out = []
    start = perf_counter()
    stop = "cap"
    text = ""
    for _ in range(policy["max_tokens"]):
        guard()
        logits = m.project(m.hidden(x, cache)[:, -1]).astype(mx.float32)
        token = (
            mx.random.categorical(logits / temperature)
            if temperature
            else logits.argmax(-1)
        ).item()
        out.append(token)
        if token in eos:
            stop = "eos"
            break
        text = t.decode(out)
        if "\n" in text.lstrip():
            stop = "line"
            break
        x = mx.array([[token]])
    response = text.lstrip().splitlines()[0] if text.strip() else ""
    return dict(
        map_id=r["map_id"],
        map=r["map"],
        start=r["start"],
        order=r["order"],
        prompt_ids=ids,
        token_ids=out,
        seed=seed,
        response=response,
        stop=stop,
        seconds=perf_counter() - start,
        **verdict(r["map"], r["start"], response, stop),
    )


def run():
    spec = read(REG)
    closed = ROOT / "decision.json"
    if closed.exists():
        return print(read(closed)["status"])
    if (ROOT / "registration.json").exists():
        raise ValueError("Prior interrupted study; preserve generations")
    start = perf_counter()
    save(ROOT / "registration.json", spec)
    assets()
    parent = Path(spec["parent"])
    if digest(parent / "adapters.safetensors") != spec["weights"]:
        raise ValueError("Unexpected parent")
    train, dev = material(spec["seed"])
    save(ROOT / "splits.json", dict(train=train, development=dev))
    m, t, _ = model_for(ARM)
    checkpoint.restore(parent, m)
    mx.reset_peak_memory()
    aborted = []

    def guard():
        if aborted:
            raise InterruptedError("Interrupted")
        if perf_counter() - start > spec["wall_seconds"] - 2:
            raise TimeoutError("Time limit")
        if (
            mx.get_peak_memory() > 6 * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            in ("2", "4")
        ):
            raise MemoryError("Resource stop")

    handlers = {
        s: signal.signal(s, lambda n, f: aborted.append(n))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    accepted = {}
    rows = []
    results = {}
    status = "incomplete"
    try:
        for r in train:
            for k in range(2):
                row = sample(m, t, r, k, spec["temperature"], spec, guard)
                rows.append(row)
                if row["verified"]:
                    accepted.setdefault(r["map_id"], row)
            packed(ROOT / "candidates.json.gz", rows)
        if len(accepted) < spec["min_verified_maps"]:
            status = "failed yield floor"
            return
        chosen = sorted(accepted)[: spec["min_verified_maps"]]
        for arm in ("self", "gold"):
            checkpoint.restore(parent, m)
            m.train()
            opt = optim.AdamW(
                learning_rate=spec["lr"], weight_decay=0, bias_correction=True
            )
            vg = nn.value_and_grad(m, loss_sum)
            root = Path("checkpoints/rsi_trace1") / arm
            root.mkdir(parents=True, exist_ok=False)
            data = []
            for key in chosen:
                q = accepted[key]
                answer = (
                    q["response"]
                    if arm == "self"
                    else ", ".join(
                        map(str, transitions.states(q["map"], q["start"], 3))
                    )
                )
                target = t.encode(answer, add_special_tokens=False) + [
                    next(iter(sorted(t.eos_token_ids)))
                ]
                data.append(
                    dict(
                        ids=q["prompt_ids"] + target,
                        prefix=len(q["prompt_ids"]),
                        id=key,
                        answer=answer,
                    )
                )
            save(ROOT / (arm + "-data.json"), data)
            history = []
            try:
                for i in range(spec["updates"]):
                    guard()
                    tick = perf_counter()
                    group = data[4 * (i % 2) : 4 * (i % 2) + 4]
                    grads = None
                    denom = 0
                    ce = 0
                    for q in group:
                        x, p, y = batch([q])
                        loss, g = vg(m, x, p, y)
                        grads = (
                            g
                            if grads is None
                            else tree_map(lambda a, b: a + b, grads, g)
                        )
                        denom += y.size
                        ce += loss.item()
                        mx.eval(grads)
                    grads, norm = optim.clip_grad_norm(
                        tree_map(lambda x: x / denom, grads), 1
                    )
                    if not math.isfinite(ce + norm.item()):
                        raise FloatingPointError("Nonfinite gradient")
                    opt.update(m, grads)
                    mx.eval(opt.state, m.trainable_parameters())
                    history.append(
                        dict(
                            targets=denom, ce=ce / denom, seconds=perf_counter() - tick
                        )
                    )
            finally:
                checkpoint.write(
                    root,
                    m,
                    opt,
                    dict(
                        step=len(history),
                        history=history,
                        parent=spec["weights"],
                        curriculum=fingerprint(data),
                    ),
                )
            result = [sample(m, t, r, 0, 0, spec, guard) for r in dev]
            packed(ROOT / (arm + "-evaluation.json.gz"), result)
            results[arm] = dict(
                verified=sum(r["verified"] for r in result),
                correct=sum(r["final_correct"] for r in result),
                weights=digest(checkpoint_location(root) / "adapters.safetensors"),
                history=history,
            )
        status = "completed"
    except (MemoryError, TimeoutError, InterruptedError, FloatingPointError) as err:
        status = str(err)
    finally:
        packed(ROOT / "candidates.json.gz", rows)
        save(
            closed,
            dict(
                status=status,
                candidates=len(rows),
                verified_maps=len(accepted),
                comparisons=results,
                seconds=perf_counter() - start,
                peak=mx.get_peak_memory(),
            ),
        )
        for s, h in handlers.items():
            signal.signal(s, h)
        print(status, len(accepted), "verified maps", flush=True)


if __name__ == "__main__":
    run()
