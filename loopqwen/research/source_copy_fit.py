"""Fit a causal pointer–generator to a frozen, two-read source-conditioned decoder."""

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
import random

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import read, save, digest
from ..shared_query_assistant import SharedQueryAssistant
from ..source_copy import SourceCopyRead
from .conditional_features import CACHE
from .core import Budget
from .file_evidence_environment import capture
from .registered import registered
from .shared_query_features import RESULTS, arrays
from .table_reentry_fit import root_for, apply, accumulate

NAME = "source-copy-fit-01"


def build(s):
    parent = SharedQueryAssistant(s["checkpoint"], RESULTS / s["parent"])._tool_model
    mx.random.seed(s["fit_seed"])
    net = SourceCopyRead(
        parent.text, revisits=s["revisits"], rank=s["rank"], gate_rank=s["gate_rank"]
    )
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    net.step.pointer.address.weight = parent.step.address.weight[...]
    net.step.freeze()
    net.step.pointer.unfreeze()
    net.eval()
    mx.eval(net.parameters())
    return net


def extract(net, w, keep_one):
    p, y = w["p"], w["y"]
    z = arrays(net, p, y)
    args = (z["pre"], z["ordinary"], z["keys"], z["values"], len(p) - 1, "causal")
    data = dict(
        values=z["values"],
        ids=mx.array([p + y[:-1]]),
        labels=mx.array([y]),
        two=net.text.model.norm(net.refine(*args, visits=2)),
    )
    if keep_one:
        data["one"] = net.text.model.norm(net.refine(*args, visits=1))
    mx.eval(data)
    return data


def features(net, rows, s, folder, guard):
    index = []
    probes = set(s["probes"])
    for i, w in enumerate(rows):
        guard.check()
        p = folder / (fingerprint([w["p"], w["y"]]) + ".safetensors")
        d = extract(net, w, w["id"] in probes)
        mx.save_safetensors(str(p), d)
        index.append(
            dict(
                id=w["id"],
                group=w["group"],
                source=w["cluster"],
                file=p.name,
                sha256=digest(p),
                positions=len(w["y"]),
                offset=len(w["p"]) - 1,
            )
        )
        if (i + 1) % 100 == 0:
            print("source-mixture features", i + 1, flush=True)
    return index


def probes(net, idx, s, folder, mode):
    want = set(s["probes"])
    out = {}
    for row in idx:
        if row["id"] not in want:
            continue
        z = mx.load(str(folder / row["file"]))
        sums = []
        for j in range(0, row["positions"], s["chunk"]):
            h = z["one" if mode == "one" else "two"][:, j : j + s["chunk"]]
            labels = z["labels"][:, j : j + s["chunk"]]
            if mode == "parent":
                p = net.project(h).astype(mx.float32)
                v = mx.take_along_axis(p, labels[..., None], axis=-1)[
                    ..., 0
                ] - mx.logsumexp(p, axis=-1)
            elif mode == "language":
                p, _, _ = net.step.pointer.distributions(
                    h, z["values"], row["offset"] + j, net.project
                )
                v = mx.take_along_axis(p, labels[..., None], axis=-1)[..., 0]
            else:
                v, _, _ = net.step.pointer.selected(
                    h, z["values"], z["ids"], row["offset"] + j, labels, net.project
                )
            sums.append(-v.sum().item())
        out[row["id"]] = sum(sums) / row["positions"]
    assert set(out) == want
    return out


def decision(idx, s, h, pr):
    buckets = defaultdict(list)
    for row in idx:
        k = row["id"]
        if k in pr["parent"]:
            buckets[row["group"]].append(k)

    def diff(k, old, new):
        return pr[old][k] - pr[new][k]

    gains = {
        g: mean(diff(k, "parent", "two") for k in keys) for g, keys in buckets.items()
    }
    copy = mean(
        mean(diff(k, "language", "two") for k in buckets[g])
        for g in ("calculate", "conversation_read")
    )
    depth = mean(mean(diff(k, "one", "two") for k in keys) for keys in buckets.values())
    gates = dict(
        updates=len(h) == 540,
        exposure=sum(v["positions"] for v in h) == s["expected_positions"],
        finite=all(
            0 < v["gradient"] < math.inf and math.isfinite(v["loss"]) for v in h
        ),
        total=mean(gains.values()) > 0.035,
        reading=gains["conversation_read"] > 0,
        action=gains["calculate"] > 0,
        quantity=gains["quantity"] > 0,
        protected=mean(gains[g] for g in ("reuse", "redirect", "retention")) >= -0.025,
        copy=copy > 0.005,
        second_read=depth > 0,
    )
    return dict(
        passed=all(gates.values()),
        gates=gates,
        gains=gains,
        copy_gain=copy,
        second_gain=depth,
        updates=len(h),
        positions=sum(v["positions"] for v in h),
    )


def fit():
    s = registered(NAME)
    root = root_for(s)
    capture(
        root,
        (
            "loopqwen/source_copy.py",
            "loopqwen/bounded_read.py",
            "loopqwen/content_read.py",
            "loopqwen/research/source_copy_fit.py",
            "scripts/register_source_copy.py",
        ),
    )
    guard = Budget(NAME, s["limits"])
    history = []
    try:
        net = build(s)
        rows = unpack(Path(s["training"]))
        folder = CACHE / NAME / digest(root / "registration.json")
        folder.mkdir(exist_ok=True, parents=True)
        idx = features(net, rows, s, folder, guard)
        packed(root / "feature-index.json.gz", idx)
        pr = {"parent": probes(net, idx, s, folder, "parent")}

        def objective(head, h, v, ids, pos, labels):
            logp, gate, _ = head.selected(h, v, ids, pos, labels, net.project)
            return -logp.sum(), gate.sum()

        grad = nn.value_and_grad(net.step.pointer, objective)
        opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)
        for epoch in range(s["epochs"]):
            order = list(range(len(idx)))
            random.Random(s["fit_seed"] + epoch).shuffle(order)
            for k in range(0, len(order), s["batch"]):
                batch = [idx[j] for j in order[k : k + s["batch"]]]
                gsum = None
                loss = 0.0
                gatesum = 0.0
                n = 0
                for r in batch:
                    guard.check()
                    z = mx.load(str(folder / r["file"]))
                    for j in range(0, r["positions"], s["chunk"]):
                        h = z["two"][:, j : j + s["chunk"]]
                        labels = z["labels"][:, j : j + s["chunk"]]
                        (v, g), gs = grad(
                            net.step.pointer,
                            h,
                            z["values"],
                            z["ids"],
                            r["offset"] + j,
                            labels,
                        )
                        weight = 1 / (len(batch) * r["positions"])
                        gsum = accumulate(gsum, gs, weight)
                        mx.eval(gsum)
                        loss += v.item() * weight
                        gatesum += g.item()
                        n += labels.size
                norm = apply(net.step.pointer, opt, gsum)
                if not math.isfinite(loss) or norm <= 0:
                    raise InterruptedError("Invalid source-mixture update")
                history.append(
                    dict(
                        epoch=epoch,
                        ids=[r["id"] for r in batch],
                        loss=loss,
                        gradient=norm,
                        gate=gatesum / n,
                        positions=n,
                    )
                )
                if len(history) % 50 == 0:
                    print("source-mixture updates", len(history), flush=True)
        for mode in ("two", "one", "language"):
            guard.check()
            pr[mode] = probes(net, idx, s, folder, mode)
        params = dict(tree_flatten(net.step.parameters()))
        prior = mx.load(str(RESULTS / s["parent"] / "step.safetensors"))
        assert all(mx.array_equal(v, params[k]).item() for k, v in prior.items())
        weights = root / "step.safetensors"
        mx.save_safetensors(str(weights), params)
        save(
            root / "fit.json",
            dict(
                weights=digest(weights),
                registration=digest(root / "registration.json"),
                backbone=read(RESULTS / s["parent"] / "fit.json")["backbone"],
                updates=len(history),
                trainable=sum(
                    v.size for _, v in tree_flatten(net.step.pointer.parameters())
                ),
            ),
        )
        save(root / "probes.json", pr)
        packed(root / "training-log.json.gz", history)
        result = decision(idx, s, history, pr)
    except (InterruptedError, MemoryError, TimeoutError) as e:
        packed(root / "partial-training-log.json.gz", history)
        result = dict(
            passed=False, updates=len(history), reason=str(e), status="interrupted"
        )
    finally:
        guard.close()
    save(root / "decision.json", result)
    print(result, flush=True)


def verify(root, result):
    s = read(root / "registration.json")
    assert s == registered(NAME)
    if "reason" in result:
        assert not result["passed"] and result["updates"] == len(
            unpack(root / "partial-training-log.json.gz")
        )
        return
    idx = unpack(root / "feature-index.json.gz")
    history = unpack(root / "training-log.json.gz")
    assert decision(idx, s, history, read(root / "probes.json")) == result
    w = read(root / "fit.json")
    assert w["weights"] == digest(root / "step.safetensors") and w[
        "registration"
    ] == digest(root / "registration.json")
    for epoch in range(s["epochs"]):
        assert sorted(
            k for b in history if b["epoch"] == epoch for k in b["ids"]
        ) == sorted(r["id"] for r in idx)


if __name__ == "__main__":
    fit()
