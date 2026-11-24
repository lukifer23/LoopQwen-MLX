"""Matched categorical studies sharing the original native optimizer and model setup."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import signal
import subprocess
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import psutil

from .. import checkpoint
from ..data import pointer5, pointer6
from ..data.tokenize import physical_batches
from ..identity import dependencies, fingerprint, training_identity
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import pointer5 as kernel


@dataclass(frozen=True)
class StudyArm:
    study: str
    name: str
    seed: int
    kind: str
    rendering: str
    objective: str
    dataset: str
    milestones: tuple
    max_depth: int

    @classmethod
    def registered(cls, name, seed, spec=pointer6.SPEC):
        kind, rendering, objective = spec["arms"][name]
        return cls(
            spec["id"],
            name,
            seed,
            kind,
            rendering,
            objective,
            spec["dataset"],
            tuple(spec["milestones"]),
            spec["trained_depth"],
        )

    @property
    def root(self):
        return Path("checkpoints") / self.study / str(self.seed) / self.name

    @property
    def output(self):
        return Path("outputs") / self.study / str(self.seed) / self.name


@lru_cache(maxsize=1)
def assets():
    expected = read("docs/measurements/phase5/base-weights-and-tokenizer.json")
    snapshot = local_snapshot()
    if any(digest(snapshot / k) != v for k, v in expected["files"].items()):
        raise ValueError("Incompatible pinned weights or tokenizer")
    return expected


def model_for(arm):
    m, t, c, info = kernel.architecture(arm.seed, arm.kind, "off")
    if arm.kind == "heads":
        m.extra_heads = [nn.Linear(1024, 8, bias=False) for _ in range(2)]
    c.update(
        name=arm.name,
        dataset=arm.dataset,
        study=arm.study,
        rendering=arm.rendering,
        objective=arm.objective,
        readout=arm.kind,
        max_depth=arm.max_depth,
    )
    return m, t, c


def heads(m, h, last):
    v = kernel.normalize(h[mx.arange(h.shape[0]), last])
    return [head(v) for head in [m.state_head, *m.extra_heads]]


def requests(group, n, arm, tok, cache):
    return kernel.request_rows(
        group, n, "multi" if arm.kind == "heads" else arm.kind, tok, cache
    )


def multi_update(m, opt):
    def loss(model, x, last, y):
        scores = heads(model, next(model.block_states(x, 1)), last)[: y.shape[1]]
        ce = mx.stack(
            [nn.losses.cross_entropy(z, y[:, j]).sum() for j, z in enumerate(scores)]
        )
        return ce.mean(), ce

    vg = nn.value_and_grad(m, loss)

    def run(group, n, cap):
        start = perf_counter()
        grads = total = None
        shapes = []
        for g in physical_batches(group, cap, next_token=False):
            x, last = kernel.inputs(g)
            y = mx.array([r["row"]["states"][:n] for r in g])
            (_, ce), grad = vg(m, x, last, y)
            grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
            total = ce if total is None else total + ce
            shapes.append(list(x.shape))
            mx.eval(total, grads)
        grads, norm = optim.clip_grad_norm(
            tree_map(lambda a: a / len(group), grads), 1.0
        )
        if not mx.isfinite(norm).item() or not mx.all(mx.isfinite(total)).item():
            raise FloatingPointError("Nonfinite multi-head update")
        opt.update(m, grads)
        mx.eval(m.trainable_parameters(), opt.state)
        pos = sum(a * b for a, b in shapes)
        return dict(
            depth=n,
            maps=len(group),
            labels=n * len(group),
            positions=pos,
            site_positions=16 * pos,
            shapes=shapes,
            ce=(total / len(group)).tolist(),
            gradient=norm.item(),
            seconds=perf_counter() - start,
            peak=mx.get_peak_memory(),
            active=mx.get_active_memory(),
            cached=mx.get_cache_memory(),
            available=psutil.virtual_memory().available,
            rss=psutil.Process().memory_info().rss,
        )

    return run


def provenance(c, arm):
    files = [
        Path(__file__),
        Path(pointer6.__file__),
        Path(kernel.__file__),
        Path(pointer5.__file__),
        Path("loopqwen/training/latent.py"),
        Path("experiments/phase5.json"),
    ]
    return dict(
        base=training_identity(c, pointer6.build()),
        assets=assets(),
        registration=pointer6.SPEC,
        arm=vars(arm),
        sources={
            str(p.name if not p.is_absolute() else p.relative_to(Path.cwd())): digest(p)
            for p in files
        },
    )


def generation(arm, n):
    return arm.root / read(arm.root / f"at-{n}.json")["directory"]


def train(arm, *, until=None, parent=None, time_budget=None):
    spec = pointer6.SPEC
    until = until or arm.milestones[-1]
    m, t, c = model_for(arm)
    opt = kernel.adam()
    ident = provenance(c, arm)
    parent_weights = None
    if parent is not None:
        source = generation(parent, 480)
        parent_weights = digest(source / "adapters.safetensors")
        checkpoint.restore(source, m, opt)
    identity = fingerprint([ident, parent_weights, time_budget])
    state = dict(
        step=0,
        history=[],
        config=c,
        identity=identity,
        provenance=ident,
        parent_weights=parent_weights,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        capacity=sum(a.size for _, a in tree_flatten(m.trainable_parameters())),
        status="running",
    )
    arm.root.mkdir(parents=True, exist_ok=True)
    if (arm.root / "latest.json").exists():
        state = checkpoint.restore(arm.root, m, opt)
        if state["identity"] != identity:
            raise ValueError("Incompatible categorical resume")

    def mark():
        p = arm.root / f"at-{state['step']}.json"
        ptr = read(arm.root / "latest.json")
        if p.exists() and read(p) != ptr:
            raise ValueError("Conflicting immutable milestone")
        if not p.exists():
            save(p, ptr)

    if state["step"] in arm.milestones or state["status"] == "completed":
        mark()
    if state["status"] == "completed" or state["step"] >= until:
        return
    fn = (
        multi_update(m, opt)
        if arm.kind == "heads"
        else kernel.make_update(
            m, opt, "terminal" if arm.objective == "outcome" else arm.kind
        )
    )
    train_rows = pointer5.rows("train")
    epoch_seen = None
    interrupted = []
    handlers = {
        sig: signal.signal(sig, lambda n, f: interrupted.append(n))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    elapsed = sum(r["seconds"] for r in state["history"])
    mx.reset_peak_memory()
    try:
        for idx in range(state["step"], until):
            absolute = idx + (480 if arm.objective == "outcome" else 0)
            epoch, offset = divmod(absolute, len(train_rows) // spec["batch"])
            if epoch != epoch_seen:
                batches = pointer6.epoch_batches(train_rows, arm.seed, epoch)
                cache = {}
                epoch_seen = epoch
            group = [
                pointer6.ordered(r, arm.rendering, arm.seed, epoch)
                for r in batches[offset]
            ]
            n = pointer6.depth(absolute, epoch)
            kernel.safe()
            if idx % 24 == 0:
                pressure = subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
                ).strip()
                if pressure in ("2", "4"):
                    raise MemoryError("System memory pressure")
            m.train()
            req = requests(group, n, arm, t, cache)
            r = fn(req, n, spec["position_cap"])
            r.update(
                ids=[x["example_id"] for x in group],
                token_digest=fingerprint([v["ids"] for v in req]),
                order_digest=fingerprint([x["order"] for x in group]),
            )
            state["history"].append(r)
            state["step"] = idx + 1
            elapsed += r["seconds"]
            complete = (
                time_budget is not None and elapsed >= time_budget
            ) or idx + 1 == arm.milestones[-1]
            if (
                complete
                or idx + 1 in arm.milestones
                or (idx + 1) % 120 == 0
                or interrupted
            ):
                state["status"] = (
                    "completed"
                    if complete
                    else "interrupted"
                    if interrupted
                    else "running"
                )
                checkpoint.write(arm.root, m, opt, state)
                if idx + 1 in arm.milestones or complete:
                    mark()
                print(arm.name, idx + 1, r["ce"], flush=True)
            if interrupted:
                raise KeyboardInterrupt("Saved the complete logical update")
            if r["peak"] > spec["max_peak_gib"] * 2**30:
                raise MemoryError("Peak exceeded registered limit")
            if complete:
                break
    except MemoryError:
        if not (arm.root / f"at-{state['step']}.json").exists():
            checkpoint.write(arm.root, m, opt, state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
