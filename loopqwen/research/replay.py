"""Use the predictions of each training forward to select a bounded replay curriculum."""

import gc
from itertools import takewhile
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from . import bootstrap, core
from .functional import summarize, orbit
from .. import checkpoint
from ..data.tokenize import physical_batches
from ..evaluation.mechanism import infer, packed
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..training import pointer5
from ..training.exposure import ARM, verify_optimizer
from ..training.mechanism import requests


def select(feedback, arm, seed):
    selected = []
    for offset in range(0, len(feedback), 3):
        slot = offset // 3 % 5

        def priority(row):
            prefix = row["prefix"]
            if arm == "static":
                rank = 0
            elif slot == 3:
                rank = prefix
            elif slot == 4:
                rank = -prefix
            else:
                rank = abs(prefix - 3)
            return rank, fingerprint([seed, row["id"]])

        selected.append(min(feedback[offset : offset + 3], key=priority)["id"])
    return selected


def updates(model, opt, s):
    def loss(m, x, last, y):
        logits = mx.stack(pointer5.outputs(m, x, last, y.shape[1], "recurrent"), axis=1)
        return nn.losses.cross_entropy(logits, y).sum(), mx.argmax(logits, axis=-1)

    grad_fn = nn.value_and_grad(model, loss)

    def step(docs):
        begin = perf_counter()
        grads = None
        observed = []
        loss_value = 0.0
        positions = 0
        for g in physical_batches(docs, s["cap"], next_token=False):
            x, last = pointer5.inputs(g)
            y = mx.array([r["row"]["states"][: s["depth"]] for r in g])
            (ce, pred), grad = grad_fn(model, x, last, y)
            grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
            mx.eval(ce, pred, grads)
            positions += x.size
            loss_value += ce.item()
            for r, p in zip(g, pred.tolist(), strict=True):
                expected = r["row"]["states"][: s["depth"]]
                prefix = len(
                    list(
                        takewhile(lambda v: v[0] == v[1], zip(p, expected, strict=True))
                    )
                )
                observed.append(dict(id=r["row"]["map_id"], predicted=p, prefix=prefix))
        n = len(docs) * s["depth"]
        grads, norm = optim.clip_grad_norm(tree_map(lambda x: x / n, grads), s["clip"])
        if not np.isfinite(loss_value + norm.item()):
            raise FloatingPointError("Nonfinite online update")
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        return dict(
            seconds=perf_counter() - begin,
            ce=loss_value / n,
            feedback=observed,
            positions=positions,
            labels=n,
        )

    return step


def evaluate(m, t, rows, s, b, path, w):
    m.eval()
    records = []
    for i in range(0, len(rows), s["batch"]):
        b.check()
        g = rows[i : i + s["batch"]]
        docs = requests(g, s["depth"], ARM, t, {})
        prob, _ = infer(m, ARM, [docs], s["depth"], False)
        for r, d, p in zip(g, docs, np.array(prob), strict=True):
            tr, cy = orbit(r["map"], r["start"])
            records.append(
                dict(
                    **r,
                    ids=d["ids"],
                    transient=tr,
                    cycle=cy,
                    probabilities=p.tolist(),
                    predicted=p.argmax(1).tolist(),
                )
            )
    packed(path, dict(weights=w, records=records))
    return dict(functional=summarize(records), anchor=bootstrap.categorical(m, t, b))


def train_round(s, b, arm, j, groups, prior_weight, parent, out):
    rows = groups[f"pool{j}"]
    cp = Path("checkpoints/research") / s["id"] / f"round{j}" / arm
    folder = out / f"round{j}" / arm
    if digest(parent / "adapters.safetensors") != prior_weight:
        raise ValueError("Unexpected parent")
    m, t = core.model(parent, s["depth"])
    opt = optim.AdamW(learning_rate=s["lr"], weight_decay=0, bias_correction=True)
    local_steps = (len(rows) + s["replay_per_round"]) // s["batch"]
    offset = (j - 1) * local_steps
    identity = fingerprint(
        [s, groups, core.assets(), prior_weight, arm, j, digest(Path(__file__))]
    )
    cp.mkdir(parents=True, exist_ok=True)
    st = dict(
        step=offset,
        identity=identity,
        parent=prior_weight,
        history=[],
        selected=[],
        selection_seconds=0.0,
    )
    if (cp / "latest.json").exists():
        st = checkpoint.restore(cp, m, opt)
        if st["identity"] != identity or st["step"] != offset + len(st["history"]):
            raise ValueError("Incompatible round cursor")
        verify_optimizer(m, opt, st["step"], s["lr"])
    elif offset:
        rec = read(parent / "recovery.json")
        gen = Path(rec["generation"])
        if (
            digest(gen / "optimizer.safetensors") != rec["optimizer_sha256"]
            or digest(gen / "adapters.safetensors") != prior_weight
        ):
            raise ValueError("Parent generation changed")
        checkpoint.restore(gen, m, opt)
        verify_optimizer(m, opt, offset, s["lr"])
    fn = updates(m, opt, s)
    docs = requests(rows, s["depth"], ARM, t, {})
    by_id = {d["row"]["map_id"]: d for d in docs}
    saved = st["step"]
    m.train()
    try:
        for idx in range(len(st["history"]), local_steps):
            b.check()
            start = idx * s["batch"]
            if start < len(rows):
                batch = docs[start : start + s["batch"]]
            else:
                if not st["selected"]:
                    begin = perf_counter()
                    feedback = [r for h in st["history"] for r in h["feedback"]]
                    st["selected"] = select(feedback, arm, s["seed"] + j)
                    st["selection_seconds"] = perf_counter() - begin
                k = start - len(rows)
                batch = [by_id[i] for i in st["selected"][k : k + s["batch"]]]
            st["history"].append(fn(batch))
            st["step"] += 1
            if st["step"] % 8 == 0:
                checkpoint.write(cp, m, opt, st)
                saved = st["step"]
    finally:
        if st["step"] != saved:
            checkpoint.write(cp, m, opt, st)
    gen = checkpoint_location(cp)
    w = digest(gen / "adapters.safetensors")
    if not (folder / "model").exists():
        bootstrap.release(gen, folder / "model")
    elif digest(folder / "model/adapters.safetensors") != w:
        raise ValueError("Milestone changed")
    result = evaluate(m, t, groups["dev"], s, b, folder / "eval.json.gz", w)
    return dict(
        **result,
        weights=w,
        parent=prior_weight,
        selected=st["selected"],
        update_seconds=sum(h["seconds"] for h in st["history"]),
        selection_seconds=st["selection_seconds"],
        labels=sum(h["labels"] for h in st["history"]),
        positions=sum(h["positions"] for h in st["history"]),
    )


def run():
    s = read("research/experiments/functional-replay-01.json")
    name = s["id"]
    out = core.OUT / name
    if core.archived_result(out, s):
        return
    b = core.Budget(name, s["limits"])
    try:
        sizes = {f"pool{j+1}": s["fresh_per_round"] for j in range(s["rounds"])}
        groups = core.reserve(
            name, s["seed"], dict(**sizes, dev=s["dev_maps"]), functional=True
        )
        parent = Path(s["parent_path"])
        w = digest(parent / "adapters.safetensors")
        if w != s["parent_sha256"]:
            raise ValueError("Wrong parent")
        m, t = core.model(parent, s["depth"])
        result = dict(
            parent=evaluate(m, t, groups["dev"], s, b, out / "initial.json.gz", w),
            rounds=[],
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        prior = {arm: (w, parent) for arm in ("static", "online")}
        for j in range(1, s["rounds"] + 1):
            arms = {}
            for arm in (("static", "online") if j % 2 else ("online", "static")):
                arms[arm] = train_round(s, b, arm, j, groups, *prior[arm], out)
                prior[arm] = (arms[arm]["weights"], out / f"round{j}" / arm / "model")
                gc.collect()
                mx.clear_cache()
            result["rounds"].append(arms)
            if any(
                (result["parent"]["anchor"][-1] - v["anchor"][-1]) / 96 > 0.05
                for v in arms.values()
            ):
                result["status"] = "anchor retention stop"
                break
        else:
            result["status"] = "completed"
        delta = (
            arms["online"]["functional"]["correct"][-1]
            - arms["static"]["functional"]["correct"][-1]
        ) / s["dev_maps"]
        result["passed"] = (
            result["status"] == "completed"
            and delta >= 0.03
            and (result["parent"]["anchor"][-1] - arms["online"]["anchor"][-1]) / 96
            <= 0.02
        )
        save(out / "decision.json", result)
    finally:
        b.close()


if __name__ == "__main__":
    run()
