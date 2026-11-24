"""Common native session budget, exact turn decoding and logical response-only updates."""

import gzip
import math
import random
import signal
import subprocess
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import psutil

from .. import checkpoint
from ..data import pointer5, pointer6
from ..data.tokenize import batch, physical_batches
from ..identity import fingerprint
from ..io import read, save, digest
from ..training.losses import loss_sum
from ..training.mechanism import StudyArm, model_for, assets
from ..training.exposure import verify_optimizer
from ..evaluation.mechanism import packed, unpack
from ..worker import acquire

RESEARCH = Path("research")
OUT = Path("outputs/research")


class Budget:
    def __init__(self, name, spec):
        self.start = time.time()
        self.name = name
        self.spec = spec
        self.last = 0
        self.interrupt = False
        state = read(RESEARCH / "state.json")
        self.session_start = datetime.fromisoformat(
            state["session_start_utc"].replace("Z", "+00:00")
        ).timestamp()
        local_session = OUT / "active-session.json"
        if local_session.exists():
            self.session_start = read(local_session)["start"]
        self.ledger = OUT / ("budget-" + str(int(self.session_start)) + ".json")
        acquire()
        self.jobs = read(self.ledger) if self.ledger.exists() else []
        limits = state["limits"]
        started = {r["job"] for r in self.jobs}
        if name not in started and len(started) >= limits["iterations"]:
            raise RuntimeError("Session iteration budget exhausted")
        spent = sum(r["seconds"] for r in self.jobs)
        already = sum(r["seconds"] for r in self.jobs if r["job"] == name)
        self.deadline = min(
            self.start + spec["wall_seconds"] - already,
            self.session_start + limits["wall_seconds"],
            self.start + limits["native_seconds"] - spent - 12,
        )
        self.old = {
            s: signal.signal(s, self.cancel) for s in (signal.SIGINT, signal.SIGTERM)
        }
        mx.reset_peak_memory()
        self.check()
        # Charge the full remaining reservation if the process exits before close().
        self.reservation = dict(
            job=name,
            seconds=self.deadline - self.start,
            start=self.start,
            status="unclosed reservation",
        )
        save(self.ledger, [*self.jobs, self.reservation])

    def cancel(self, *_):
        self.interrupt = True

    def check(self):
        now = time.time()
        if self.interrupt:
            raise InterruptedError("Stop requested")
        if now >= self.deadline - 2:
            raise TimeoutError("Execution budget")
        if mx.get_peak_memory() > self.spec["peak_gib"] * 2**30:
            raise MemoryError("MLX peak")
        if now - self.last >= 0.7:
            self.last = now
            pressure = subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            if (
                pressure != "1"
                or psutil.virtual_memory().available
                < self.spec["available_gib"] * 2**30
            ):
                raise MemoryError("System memory pressure")

    def close(self):
        self.jobs.append(
            dict(
                job=self.name,
                seconds=time.time() - self.start,
                peak_bytes=mx.get_peak_memory(),
                start=self.start,
            )
        )
        save(self.ledger, self.jobs)
        for s, h in self.old.items():
            signal.signal(s, h)


def reserve(name, seed, sizes, functional=False):
    manifest = RESEARCH / "data/historical.json"
    source = RESEARCH / "data/historical-maps.txt.gz"
    if digest(source) != read(manifest)["sha256"]:
        raise ValueError("Historical inventory digest")
    root = RESEARCH / "data"
    path = root / (name + ".json.gz")
    key = fingerprint([seed, sizes, functional, digest(source)])
    if path.exists():
        index = RESEARCH / "registry.json"
        if index.exists():
            published = read(index)["datasets"].get(str(path))
            if published and digest(path) != published["sha256"]:
                raise ValueError("Reserved data file changed")
        saved = unpack(path)
        if saved["key"] != key:
            raise ValueError("Data registration changed")
        return saved["splits"]
    excluded = set(gzip.decompress(source.read_bytes()).decode().splitlines())
    for p in root.glob("*.json.gz"):
        for group in unpack(p)["splits"].values():
            excluded.update(r["map_id"] for r in group)
    rng = random.Random(seed)
    splits = {}
    for split, count in sizes.items():
        rows = []
        for _ in range(200000):
            mapping = (
                [rng.randrange(8) for _ in range(8)]
                if functional
                else rng.sample(range(8), 8)
            )
            r = pointer5.instance(mapping, rng.randrange(8))
            if r["map_id"] in excluded or (not functional and pointer5.cycle(r) < 4):
                continue
            excluded.add(r["map_id"])
            rows.append(pointer6.ordered(r, "random0", seed))
            if len(rows) == count:
                break
        if len(rows) != count:
            raise RuntimeError("Map space exhausted")
        splits[split] = rows
    packed(path, dict(key=key, splits=splits))
    return splits


def suffix_adapters(model):
    from ..training.adapters import attach

    mx.random.seed(109)
    attach(model, "last8", rank=8, alpha=16)


def model(path, depth):
    assets()
    arm = StudyArm(
        "research",
        "trace",
        109,
        "recurrent",
        "randomized",
        "response_ce",
        "research",
        (),
        3,
    )
    composition = Path(path) / "composition.json" if path else None
    if composition and composition.exists():
        meta = read(composition)
        parent = Path(meta["frozen_parent"])
        if digest(parent / "adapters.safetensors") != meta["sha256"]:
            raise ValueError("Frozen component differs")
        m, t = model(parent, depth)
        suffix_adapters(m)
    else:
        m, t, _ = model_for(arm)
    if path:
        checkpoint.restore(path, m)
    m.span = replace(m.span, passes=depth)
    m.eval()
    return m, t


def atom(t, text):
    ids = t.encode(text, add_special_tokens=False)
    if len(ids) != 1 or t.decode(ids) != text:
        raise ValueError("Control token changed: " + text)
    return ids[0]


def generate(
    m, t, ids, cap, guard, *, line=False, thinking=False, temperature=0, seed=0
):
    """Terminate on the first generated boundary. Preserve the complete charged token stream."""
    end = atom(t, "<|im_end|>")
    opening = atom(t, "<think>")
    closing = atom(t, "</think>")
    if thinking:
        last_open = max((i for i, x in enumerate(ids) if x == opening), default=-1)
        last_close = max((i for i, x in enumerate(ids) if x == closing), default=-1)
        if last_open <= last_close:
            raise ValueError("Expected an open thinking turn")
    mx.random.seed(seed)
    m.eval()
    cache = m.make_cache()
    x = mx.array([ids])
    out = []
    start = time.perf_counter()
    stop = "cap"
    boundary = False
    visible = ""
    close_at = None
    bad = False
    for i in range(cap):
        guard()
        while x.shape[1] > 128:
            mx.eval(m.hidden(x[:, :128], cache))
            x = x[:, 128:]
        z = m.project(m.hidden(x, cache)[:, -1]).astype(mx.float32)
        tok = (
            mx.random.categorical(z / temperature) if temperature else z.argmax(-1)
        ).item()
        if i == 0:
            ttft = time.perf_counter() - start
        out.append(tok)
        if tok == end or tok in t.eos_token_ids:
            boundary = True
            stop = "chat_end" if tok == end else "eos"
            break
        if thinking and tok == opening:
            bad = True
        if thinking and tok == closing:
            if close_at is not None:
                bad = True
            close_at = i
        if line and "\n" in t.decode(out).lstrip():
            stop = "line"
            break
        x = mx.array([[tok]])
    content = out[:-1] if boundary else out
    if not thinking:
        visible = t.decode(content)
    elif close_at is not None and not bad:
        visible = t.decode(content[close_at + 1 :])
    return dict(
        prompt_ids=ids,
        token_ids=out,
        text=t.decode(out),
        visible=visible,
        stop=stop,
        close_at=close_at,
        complete=stop != "cap" and (not thinking or close_at is not None and not bad),
        ttft=ttft,
        seconds=time.perf_counter() - start,
    )


def doc(t, prompt, response):
    pre = t.encode(prompt, add_special_tokens=False)
    all_ids = t.encode(prompt + response, add_special_tokens=False)
    if all_ids[: len(pre)] != pre or t.decode(all_ids) != prompt + response:
        raise ValueError("Response boundary merged")
    return dict(ids=all_ids + [atom(t, "<|im_end|>")], prefix=len(pre))


def state_loss(model, inputs, last, labels):
    from ..training.pointer5 import readout

    losses = [
        nn.losses.cross_entropy(readout(model, h, last), labels[:, j]).sum()
        for j, h in enumerate(model.block_states(inputs, labels.shape[1]))
    ]
    return mx.stack(losses).sum()


def train(m, docs, spec, root, budget, identity, *, state_docs=None):
    size = spec["logical_batch"]
    n = spec["updates"]
    cap = spec["cap_positions"]
    if bool(state_docs) != bool(spec.get("state_weight", 0)):
        raise ValueError("Registered state objective and supplied supervision differ")
    if state_docs is not None and len(state_docs) != len(docs):
        raise ValueError("Unpaired state supervision")
    if len(docs) != size * n:
        raise ValueError("Schedule and logical budget differ")
    key = fingerprint(
        [
            identity,
            spec,
            [r["ids"] for r in docs],
            [r["prefix"] for r in docs],
            state_docs if state_docs else [],
            assets(),
            digest(Path(__file__)),
            digest(Path(__file__).with_name("bootstrap.py")),
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    opt = optim.AdamW(
        learning_rate=spec["lr"],
        weight_decay=spec["weight_decay"],
        bias_correction=True,
    )
    st = dict(identity=key, spec=spec, parent=identity, step=0, history=[])
    if (root / "latest.json").exists():
        st = checkpoint.restore(root, m, opt)
        if st["identity"] != key:
            raise ValueError("Incompatible resume")
        if st["step"]:
            verify_optimizer(m, opt, st["step"], spec["lr"])
    m.train()
    vg = nn.value_and_grad(m, loss_sum)
    state_vg = nn.value_and_grad(m, state_loss) if state_docs else None
    saved = st["step"]
    try:
        for idx in range(saved, n):
            budget.check()
            begin = time.perf_counter()
            group = docs[idx * size : (idx + 1) * size]
            grads = None
            total = 0.0
            denom = positions = 0
            for g in physical_batches(group, cap):
                x, p, y = batch(g)
                loss, grad = vg(m, x, p, y)
                grads = (
                    grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                )
                mx.eval(grads, loss)
                total += loss.item()
                denom += y.size
                positions += x.size
            grads = tree_map(lambda a: a / denom, grads)
            state_total = 0.0
            state_count = state_positions = 0
            if state_docs:
                from ..training.pointer5 import inputs

                state_grads = None
                state_group = state_docs[idx * size : (idx + 1) * size]
                for g in physical_batches(state_group, cap, next_token=False):
                    x, last = inputs(g)
                    labels = mx.array([r["states"] for r in g])
                    loss, grad = state_vg(m, x, last, labels)
                    state_grads = (
                        grad
                        if state_grads is None
                        else tree_map(lambda a, b: a + b, state_grads, grad)
                    )
                    mx.eval(state_grads, loss)
                    state_total += loss.item()
                    state_count += labels.size
                    state_positions += x.size
                weight = spec["state_weight"]
                grads = tree_map(
                    lambda a, b: (1 - weight) * a + weight * b / state_count,
                    grads,
                    state_grads,
                )
            grads, norm = optim.clip_grad_norm(grads, spec["clip"])
            if not math.isfinite(total + state_total + norm.item()):
                raise FloatingPointError("Nonfinite update")
            opt.update(m, grads)
            mx.eval(m.trainable_parameters(), opt.state)
            st["step"] += 1
            st["history"].append(
                dict(
                    ce=total / denom,
                    state_ce=state_total / state_count if state_count else None,
                    state_labels=state_count,
                    state_positions=state_positions,
                    targets=denom,
                    positions=positions,
                    input_tokens=sum(len(r["ids"]) - 1 for r in group),
                    seconds=time.perf_counter() - begin,
                    peak=mx.get_peak_memory(),
                )
            )
            if st["step"] % 16 == 0:
                checkpoint.write(root, m, opt, st)
                saved = st["step"]
    finally:
        if saved != st["step"]:
            checkpoint.write(root, m, opt, st)
    m.eval()
    return st
