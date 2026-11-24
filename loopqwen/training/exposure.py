"""Exact continuation from a verified optimizer generation with an absolute cursor."""

import signal
import subprocess
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import psutil

from .. import checkpoint
from ..data import pointer5, pointer6
from ..identity import fingerprint
from ..io import digest, read, save
from . import pointer5 as kernel
from .mechanism import StudyArm, generation, model_for, provenance, requests

REG = Path("experiments/phase7.json")
S = read(REG)
PARENT = StudyArm.registered(S["parent"]["arm"], S["seed"])
ARM = StudyArm(
    S["id"],
    S["arm"],
    S["seed"],
    "recurrent",
    S["rendering"],
    S["objective"],
    S["dataset"],
    tuple(S["milestones"]),
    S["depth"],
)
ARCH = Path("docs/measurements/phase6")
OUT = ARM.output


def parent_lineage():
    p = S["parent"]
    g = generation(PARENT, p["step"])
    hashes = {
        key: digest(g / file)
        for key, file in (
            ("metadata", "state.json"),
            ("weights", "adapters.safetensors"),
            ("optimizer", "optimizer.safetensors"),
        )
    }
    if any(hashes[k] != p[k] for k in hashes):
        raise ValueError("Parent generation differs from registered recovery state")
    state = read(g / "state.json")
    pub = read(ARCH / "randomized/state.json")
    if (
        {k: v for k, v in state.items() if k != "history"} != pub
        or state["status"] != "completed"
        or state["step"] != p["step"]
    ):
        raise ValueError("Parent metadata differs from publication")
    if fingerprint(provenance(state["config"], PARENT)) != fingerprint(
        state["provenance"]
    ):
        raise ValueError("Incompatible parent implementation or assets")
    return dict(
        **hashes, path=str(g), identity=state["identity"], absolute_step=state["step"]
    )


def verify_optimizer(model, opt, step, lr):
    state = dict(tree_flatten(opt.state))
    params = dict(tree_flatten(model.trainable_parameters()))
    expected = {
        k + "." + moment: v.shape for k, v in params.items() for moment in ("m", "v")
    }
    if set(state) != set(expected) | {"step", "learning_rate"} or any(
        state[k].shape != shape for k, shape in expected.items()
    ):
        raise ValueError("Incompatible AdamW moment schema")
    if state["step"].item() != step or abs(state["learning_rate"].item() - lr) > 1e-10:
        raise ValueError("AdamW cursor or learning rate mismatch")


def resume_check(state, identity):
    if (
        state["identity"] != identity
        or state["step"] != state["lineage"]["absolute_step"] + len(state["history"])
        or state["step"] != state["next_absolute_step"]
    ):
        raise ValueError("Incompatible child continuation")


def train_until(end):
    lineage = parent_lineage()
    start = lineage["absolute_step"]
    if not start < end <= start + S["max_child_updates"]:
        raise ValueError("Unregistered continuation boundary")
    model, t, c = model_for(ARM)
    for actual, key in (
        (c["rank"], "rank"),
        (c["lora_alpha"], "scale"),
        (model.checkpointing, "checkpointing"),
        (kernel.SPEC["clip"], "clip"),
        (pointer6.SPEC["batch"], "batch"),
    ):
        if actual != S[key]:
            raise ValueError(f"Unsupported registered policy: {key}")
    provenance_data = provenance(c, ARM)
    identity = fingerprint([S, lineage, provenance_data, digest(Path(__file__))])
    opt = optim.AdamW(
        learning_rate=S["lr"], weight_decay=S["decay"], bias_correction=True
    )
    ARM.root.mkdir(parents=True, exist_ok=True)
    if (ARM.root / "latest.json").exists():
        state = checkpoint.restore(ARM.root, model, opt)
        resume_check(state, identity)
    else:
        checkpoint.restore(lineage["path"], model, opt)
        state = dict(
            step=start,
            next_absolute_step=start,
            child_completed_updates=0,
            lineage=lineage,
            identity=identity,
            policy=S,
            config=c,
            provenance=provenance_data,
            history=[],
            wall_seconds=0,
            status="running",
            git=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        )
    verify_optimizer(model, opt, state["step"], S["lr"])
    if state["step"] >= end or state["status"] in ("time_limit", "completed"):
        return state
    model.train()
    fn = kernel.make_update(model, opt, "recurrent")
    rows = pointer5.rows("train")
    batches = pointer6.epoch_batches(
        rows, ARM.seed, state["step"] // (len(rows) // S["batch"])
    )
    interrupted = []
    handlers = {
        sig: signal.signal(sig, lambda n, f: interrupted.append(n))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    cache = {}
    elapsed = sum(x["seconds"] for x in state["history"])
    clock = perf_counter()
    mx.reset_peak_memory()
    try:
        while state["step"] < end:
            absolute = state["step"]
            epoch, i = divmod(absolute, len(batches))
            if (
                elapsed
                + max([2.0] + [r["seconds"] * 2 for r in state["history"][-16:]])
                >= S["max_update_seconds"]
            ):
                state["status"] = "time_limit"
                break
            if psutil.virtual_memory().available < S["available_gib"] * 2**30 or (
                i % 24 == 0
                and subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
                ).strip()
                in ("2", "4")
            ):
                raise MemoryError("Registered memory limit reached")
            group = [
                pointer6.ordered(r, ARM.rendering, ARM.seed, epoch) for r in batches[i]
            ]
            n = pointer6.depth(absolute, epoch)
            req = requests(group, n, ARM, t, cache)
            r = fn(req, n, S["cap"])
            r.update(
                ids=[v["example_id"] for v in group],
                order_digest=fingerprint([v["order"] for v in group]),
                token_digest=fingerprint([v["ids"] for v in req]),
                epoch=epoch,
                absolute_index=absolute,
            )
            state["history"].append(r)
            elapsed += r["seconds"]
            state.update(
                step=absolute + 1,
                next_absolute_step=absolute + 1,
                child_completed_updates=absolute + 1 - start,
                optimizer_step=int(opt.step.item()),
            )
            if state["optimizer_step"] != state["step"]:
                raise RuntimeError("Optimizer cursor drift")
            if r["peak"] > S["peak_gib"] * 2**30:
                raise MemoryError("MLX peak exceeded")
            if interrupted:
                state["status"] = "interrupted"
                break
            if state["step"] % 120 == 0 and state["step"] < end:
                state["wall_seconds"] += perf_counter() - clock
                checkpoint.write(ARM.root, model, opt, state)
                clock = perf_counter()
        else:
            state["status"] = "completed" if end == ARM.milestones[-1] else "milestone"
    except (MemoryError, FloatingPointError):
        state["status"] = "unsafe"
        raise
    finally:
        state["wall_seconds"] += perf_counter() - clock
        checkpoint.write(ARM.root, model, opt, state)
        pointer = ARM.root / f"at-{state['step']}.json"
        if not pointer.exists():
            save(pointer, read(ARM.root / "latest.json"))
        for sig, h in handlers.items():
            signal.signal(sig, h)
    if interrupted:
        raise KeyboardInterrupt
    return state
