"""Fit the two shared content-read matrices from original source locations and actual calls."""

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from .. import checkpoint
from ..content_read import ContentRead
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..shared_query import SharedQuery
from ..shared_query_assistant import SharedQueryAssistant
from .conditional_features import CACHE
from .core import Budget
from .file_evidence_environment import capture
from .learner_prefix_fit import extract, logprob
from .registered import registered
from .shared_query_features import RESULTS, SOURCES
from .shared_query_fit import parts, schedule, cited_loss
from .table_reentry_fit import apply, accumulate, root_for

NAME = "content-read-fit-01"


def build(s):
    parent = SharedQueryAssistant(s["checkpoint"], RESULTS / s["parent"])._tool_model
    mx.random.seed(s["fit_seed"])
    model_class = ContentRead
    if s.get("bounded_write"):
        from ..bounded_read import BoundedContentRead

        model_class = BoundedContentRead
    net = model_class(
        parent.text, revisits=s["revisits"], rank=s["rank"], gate_rank=s["gate_rank"]
    )
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    # Keep every existing controller tensor exactly at its archived value.
    net.step.freeze()
    net.step.address.unfreeze()
    net.step.write.unfreeze()
    net.eval()
    mx.eval(net.step.parameters())
    assert set(dict(tree_flatten(net.step.trainable_parameters()))) == {
        "address.weight",
        "write.weight",
    }
    return net


def prediction(net, step, args, mode, grad=False):
    pre, h, k, v, pos, y, marks = args
    mask = "causal" if y.size > 1 else None
    if mode == "parent":
        state = SharedQuery.refine(net, pre, h, k, v, pos, mask, step=step, visits=2)
        queries = []
    else:
        queries = []
        state = net.refine(
            pre,
            h,
            k,
            v,
            pos,
            mask,
            step=step,
            visits=1 if mode == "one" else 2,
            differentiate=grad,
            observe_content=queries.append,
        )
    logp = logprob(net.project(net.text.model.norm(state)).astype(mx.float32))
    nll = -mx.take_along_axis(logp, y[None, :, None], axis=-1).sum()
    ev = (
        cited_loss(queries[0], v, pos, marks, net.step.dimension**-0.5)
        if marks and queries
        else mx.array(0.0)
    )
    return logp, nll, ev


def evidence_for(rows):
    return {
        w["id"]: dict(
            prefix=fingerprint(w["p"]),
            labels=fingerprint(w["y"]),
            evidence=w["annotations"],
        )
        for w in rows
        if w["annotations"]
    }


def probes(net, s, idx, folder, ev, mode, guard=None):
    selected = set(s["probes"])
    res = {}
    for r in idx:
        if r["id"] not in selected:
            continue
        nll = 0.0
        citation = 0.0
        count = 0
        for args in parts(r, folder, ev, s):
            if guard:
                guard.check()
            _, loss, e = prediction(net, net.step, args, mode)
            nll += loss.item()
            citation += e.item()
            count += len(args[-1])
        res[r["id"]] = dict(
            nll=nll / r["positions"], citation_nll=citation, marks=count
        )
    assert selected == set(res)
    return res


def decide(s, idx, h, p):
    groups = defaultdict(list)
    extra = []
    depth = []
    before = []
    after = []
    marks = 0
    identity = []
    for r in idx:
        k = r["id"]
        if k not in p["before"]:
            continue
        initial = p["before"][k]
        final = p["trained"][k]
        base = p["parent"][k]
        one = p["one"][k]
        identity.append(abs(base["nll"] - initial["nll"]))
        groups[r["group"].removesuffix("/source")].append(initial["nll"] - final["nll"])
        extra.append(base["nll"] - final["nll"])
        depth.append(one["nll"] - final["nll"])
        before.append(initial["citation_nll"])
        after.append(final["citation_nll"])
        marks += final["marks"]
    gains = {k: mean(x) for k, x in groups.items()}
    protected = mean(
        v
        for k, v in gains.items()
        if k not in ("reasoning", "calculate", "quantity", "reuse", "redirect")
    )
    evidence = (sum(before) - sum(after)) / marks
    gates = dict(
        identity=max(identity) < 1e-6,
        exposure=sum(w["positions"] for w in h) == s["expected_positions"]
        and sum(w["marks"] for w in h) == s["expected_citations"],
        finite=all(
            0 < w["gradient"] < math.inf and math.isfinite(w["loss"]) for w in h
        ),
        reasoning=gains["reasoning"] > 0.02,
        action=gains["calculate"] > 0.02,
        quantity=gains["quantity"] > 0.02,
        database=mean([gains["reuse"], gains["redirect"]]) > 0,
        protected=protected >= -0.05,
        evidence=evidence >= 0.1,
        content=mean(extra) > 0.01,
        second_read=mean(depth) > 0.002,
    )
    if s.get("bounded_write"):
        gates["early_parent_divergence"] = max(w["kl"] for w in h[:16]) < 0.5
    return dict(
        passed=all(gates.values()),
        gates=gates,
        gains=gains,
        max_initial_error=max(identity),
        citation_gain=evidence,
        protected_gain=protected,
        content_gain=mean(extra),
        second_gain=mean(depth),
        updates=len(h),
        positions=sum(r["positions"] for r in h),
        citations=sum(r["marks"] for r in h),
    )


def fit(name=NAME):
    s = registered(name)
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/content_read.py",
            *(("loopqwen/bounded_read.py",) if s.get("bounded_write") else ()),
            "loopqwen/research/content_read_fit.py",
            "loopqwen/research/learner_prefix_fit.py",
            "loopqwen/research/shared_query_fit.py",
            "scripts/register_content_read.py",
        ),
    )
    guard = Budget(name, s["limits"])
    history = []
    try:
        net = build(s)
        rows = unpack(Path(s["training"]))
        ev = evidence_for(rows)
        work = CACHE / name / digest(root / "registration.json")
        work.mkdir(parents=True, exist_ok=True)
        folder = work
        folder.mkdir(parents=True, exist_ok=True)
        if s.get("feature_source"):
            source = RESULTS / s["feature_source"]
            folder = CACHE / s["feature_source"] / digest(source / "registration.json")
            idx = unpack(source / "feature-index.json.gz")
            for r, w in zip(idx, rows, strict=True):
                guard.check()
                assert r["id"] == w["id"] and r["sha256"] == digest(folder / r["file"])
                assert (r["prefix"], r["labels"]) == (
                    fingerprint(w["p"]),
                    fingerprint(w["y"]),
                )
        else:
            idx = extract(net, rows, folder, guard)
        packed(root / "feature-index.json.gz", idx)
        p = {m: probes(net, s, idx, folder, ev, m, guard) for m in ("before", "parent")}
        if any(
            abs(p["before"][k]["nll"] - p["parent"][k]["nll"]) >= 1e-6
            for k in p["before"]
        ):
            raise InterruptedError("Initial content read changed the trained parent")
        opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)

        def loss(step, parent, *args):
            logp, nll, e = prediction(net, step, args, "trained", True)
            kl = (mx.exp(parent) * (parent - logp)).sum()
            objective = nll + s["teacher_kl"] * kl + s["evidence_weight"] * e
            return objective / args[-2].size, nll, e, kl

        fn = nn.value_and_grad(net.step, loss)
        steps = schedule(idx, s)
        for j, (epoch, ids) in enumerate(steps):
            total = sum(idx[i]["positions"] for i in ids)
            grad = None
            metrics = mx.zeros((4,))
            marks = 0
            for i in ids:
                for args in parts(idx[i], folder, ev, s):
                    guard.check()
                    parent = prediction(net, net.step, args, "parent")[0]
                    mx.eval(parent)
                    (obj, nll, e, kl), g = fn(net.step, parent, *args)
                    weight = args[-2].size / total
                    grad = accumulate(grad, g, weight)
                    metrics = metrics + mx.stack(
                        [weight * obj, nll / total, e / total, kl / total]
                    )
                    marks += len(args[-1])
                    mx.eval(grad, metrics)
            norm = apply(net.step, opt, grad)
            loss_val, nll, e, kl = metrics.tolist()
            if (
                not all(math.isfinite(v) for v in (norm, loss_val, nll, e, kl))
                or norm <= 0
            ):
                raise InterruptedError("Nonfinite or zero update")
            history.append(
                dict(
                    epoch=epoch,
                    rows=[idx[i]["id"] for i in ids],
                    positions=total,
                    marks=marks,
                    gradient=norm,
                    loss=loss_val,
                    nll=nll,
                    evidence=e,
                    kl=kl,
                )
            )
            if s.get("bounded_write") and j < 16 and kl >= 0.5:
                raise InterruptedError(
                    "Early parent divergence reached the registered limit"
                )
            if j + 1 == len(steps) or steps[j + 1][0] != epoch:
                checkpoint.write(
                    work,
                    net.step,
                    opt,
                    dict(
                        step=j + 1,
                        epoch=epoch,
                        registration=digest(root / "registration.json"),
                        history=history,
                    ),
                )
            if (j + 1) % 100 == 0:
                print("updates", j + 1, "of", len(steps), flush=True)
        for m in ("trained", "one"):
            p[m] = probes(net, s, idx, folder, ev, m, guard)
        params = dict(tree_flatten(net.step.parameters()))
        previous = mx.load(str(RESULTS / s["parent"] / "step.safetensors"))
        assert all(mx.array_equal(v, params[k]).item() for k, v in previous.items())
        weights = root / "step.safetensors"
        mx.save_safetensors(str(weights), params)
        save(
            root / "fit.json",
            dict(
                weights=digest(weights),
                registration=digest(root / "registration.json"),
                backbone=read(RESULTS / s["parent"] / "fit.json")["backbone"],
                updates=len(history),
                trainable=sum(v.size for k, v in params.items() if k not in previous),
            ),
        )
        packed(root / "training-log.json.gz", history)
        save(root / "probes.json", p)
        decision = decide(s, idx, history, p)
    except (InterruptedError, MemoryError, TimeoutError) as e:
        packed(root / "partial-training-log.json.gz", history)
        decision = dict(
            passed=False, status="interrupted", reason=str(e), updates=len(history)
        )
    finally:
        guard.close()
    save(root / "decision.json", decision)
    print(decision, flush=True)


def verify(root, d):
    s = registered(read(root / "registration.json")["id"])
    assert s == read(root / "registration.json")
    if d.get("status") == "interrupted":
        assert (
            not d["passed"]
            and d["reason"]
            and len(unpack(root / "partial-training-log.json.gz")) == d["updates"]
        )
        return
    idx = unpack(root / "feature-index.json.gz")
    history = unpack(root / "training-log.json.gz")
    for r, w in zip(idx, unpack(Path(s["training"])), strict=True):
        assert (r["prefix"], r["labels"], r["id"]) == (
            fingerprint(w["p"]),
            fingerprint(w["y"]),
            w["id"],
        )
    for r, (epoch, batch) in zip(history, schedule(idx, s), strict=True):
        assert r["epoch"] == epoch and [idx[i]["id"] for i in batch] == r["rows"]
    assert d == decide(s, idx, history, read(root / "probes.json"))
    f = read(root / "fit.json")
    assert (
        f["weights"] == digest(root / "step.safetensors")
        and f["registration"] == digest(root / "registration.json")
        and f["backbone"] == read(RESULTS / s["parent"] / "fit.json")["backbone"]
    )


if __name__ == "__main__":
    fit()
