import fcntl
import hashlib
import math
import random
import signal
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import psutil

from .. import checkpoint
from ..config import config, span
from ..data.build_data import ROOT, load_split
from ..data.tokenize import batch, pack
from ..hardware import doctor
from ..io import read, save
from ..model_loader import count
from .losses import loss_sum, packed_loss_sum
from .optimizers import optimizer


def evaluate_rows(model, rows, response_only=True):
    model.eval()
    total = tokens = 0
    for row in rows:
        x, pos, y = batch([row], response_only)
        total += loss_sum(model, x, pos, y).item()
        tokens += y.size
    return dict(nll=total / tokens, tokens=tokens, perplexity=math.exp(total / tokens))


def depth(c, step, micro):
    rng = random.Random(f"{c['seed']}/{step}/{micro}")
    if c["curriculum"] == "mixed":
        return rng.choices((1, 2, 3), weights=c["depth_weights"])[0]
    if c["curriculum"] == "gradual" and rng.random() >= min(
        1.0, step / max(1, c["steps"] / 3)
    ):
        return 1
    return c["passes"]


def train(path, until=None):
    c = config(path)
    root = Path("checkpoints") / c["name"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(c, root, until)


def _train(c, root, until):
    manifest = (
        read(ROOT / "manifest.json") if (ROOT / "manifest.json").exists() else None
    )
    rows = load_split(c["training_split"])
    random.Random(c["seed"]).shuffle(rows)
    valid = load_split(c["validation_split"])[: c["eval_count"]]
    model, tok, info = checkpoint.setup(c)
    opt = optimizer(c)
    packed = c.get("packing", False)
    fn = nn.value_and_grad(model, packed_loss_sum if packed else loss_sum)
    state = dict(
        config=c,
        manifest=manifest,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hardware=doctor(),
        model=info,
        trainable=count(model, True),
        step=0,
        cursor=0,
        history=[],
        input_tokens=0,
        loss_tokens=0,
        order_hash="",
        sequence_length=c["sequence_length"],
        batch_size=c["batch_size"],
        checkpointing=c["gradient_checkpointing"],
    )
    if (root / "latest.json").exists():
        state = checkpoint.restore(root, model, opt)
        if state["config"] != c or state["manifest"] != manifest:
            raise ValueError("Resume config/data mismatch. Use a distinct run name.")
    stop = c["steps"] if until is None else min(c["steps"], until)
    if stop < state["step"]:
        raise ValueError("Requested stop precedes checkpoint")
    canceled = []

    def request_stop(signum, frame):
        canceled.append(signum)

    handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    mx.set_cache_limit(384 * 1024**2)
    mx.reset_peak_memory()
    begin = perf_counter()
    try:
        while state["step"] < stop:
            s = state["step"]
            pressure = (
                psutil.virtual_memory().available < c["min_available_gib"] * 2**30
                or mx.get_peak_memory() > c["memory_target_gib"] * 2**30
            )
            if pressure:
                mx.clear_cache()
                mx.reset_peak_memory()
                if state["sequence_length"] > 16:
                    state["sequence_length"] = max(16, state["sequence_length"] // 2)
                elif state["batch_size"] > 1:
                    state["batch_size"] = max(1, state["batch_size"] // 2)
                elif not state["checkpointing"]:
                    state["checkpointing"] = True
                else:
                    raise MemoryError(
                        "At minimum batch and length under configured memory pressure. Resume after freeing memory."
                    )
            model.checkpointing = state["checkpointing"]
            model.train()
            mx.random.seed(c["seed"] + s)
            if c["alpha_schedule"] is not None:
                a, b = c["alpha_schedule"]
                model.alpha_override = a + (b - a) * s / max(1, c["steps"] - 1)
            total = grads = None
            labels = inputs = 0
            ids = []
            depths = []
            cursor = state["cursor"]
            start = perf_counter()
            for micro in range(c["accumulation"]):
                selected = []
                for _ in range(len(rows) * state["batch_size"]):
                    row = rows[cursor % len(rows)]
                    cursor += 1
                    if len(row["ids"]) - 1 <= state["sequence_length"]:
                        selected.append(row)
                    if len(selected) == state["batch_size"]:
                        break
                if len(selected) != state["batch_size"]:
                    raise ValueError("Sequence length too short for dataset")
                k = depth(c, s, micro)
                model.span = replace(span(c), passes=k)
                depths.append(k)
                arguments = (pack if packed else batch)(selected, c["response_only"])
                value, g = fn(model, *arguments)
                y = arguments[2]
                grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
                total = value if total is None else total + value
                mx.eval(grads, total)
                labels += y.size
                inputs += sum(len(r["ids"]) - 1 for r in selected)
                ids.extend(r["id"] for r in selected)
            grad_seconds = perf_counter() - start
            grads = tree_map(lambda g: g / labels, grads)
            grads, norm = optim.clip_grad_norm(grads, c["clip"])
            mean = total.item() / labels
            magnitude = norm.item()
            if not math.isfinite(mean) or not math.isfinite(magnitude):
                raise FloatingPointError(f"Nonfinite loss or gradient at step {s + 1}")
            t = perf_counter()
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            elapsed = perf_counter() - start
            row = dict(
                step=s + 1,
                loss=mean,
                gradient_norm=magnitude,
                inputs=inputs,
                labels=labels,
                depths=depths,
                seconds=elapsed,
                grad_seconds=grad_seconds,
                optimizer_s=perf_counter() - t,
                peak_bytes=mx.get_peak_memory(),
                sequence_limit=state["sequence_length"],
                batch_size=state["batch_size"],
            )
            state.update(
                step=s + 1,
                cursor=cursor,
                input_tokens=state["input_tokens"] + inputs,
                loss_tokens=state["loss_tokens"] + labels,
                order_hash=hashlib.sha256(
                    (state["order_hash"] + "|".join(ids)).encode()
                ).hexdigest(),
                alpha_override=model.alpha_override,
                next_mlx_seed=c["seed"] + s + 1,
            )
            model.span = span(c)
            if (s + 1) % c["eval_interval"] == 0 or s + 1 == c["steps"]:
                row["validation"] = evaluate_rows(model, valid, c["response_only"])
            state["history"].append(row)
            if (s + 1) % c["checkpoint_interval"] == 0 or s + 1 == stop or canceled:
                checkpoint.write(root, model, opt, state)
                save(Path("outputs") / (c["name"] + ".json"), state)
                print(
                    f"{c['name']} update {s + 1} loss {mean:.4f}, {inputs / elapsed:.1f} input tok/s, peak {row['peak_bytes'] / 2**30:.2f} GiB",
                    flush=True,
                )
            if canceled:
                raise KeyboardInterrupt("Saved at update boundary")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return dict(
        run=c["name"],
        step=state["step"],
        input_tokens=state["input_tokens"],
        loss_tokens=state["loss_tokens"],
        wall_seconds=perf_counter() - begin,
    )
