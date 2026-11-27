"""Correct source-dependent continuations after the recurrent decoder's own decisions."""

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from .. import checkpoint
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..shared_query import QueryStep
from ..shared_query_assistant import SharedQueryAssistant
from .conditional_features import CACHE
from .core import Budget
from .file_evidence_environment import capture
from .registered import registered
from .shared_query_features import RESULTS, SOURCES, arrays
from .shared_query_fit import parts, schedule
from .table_reentry_fit import accumulate, apply, root_for


def extract(net, rows, folder, guard):
    index = []
    for i, w in enumerate(rows):
        guard.check()
        p, y = w["p"], w["y"]
        key = fingerprint([p, y])
        path = folder / (key + ".safetensors")
        if not path.exists():
            data = arrays(net, p, y)
            mx.eval(data)
            tmp = path.with_suffix(".new.safetensors")
            mx.save_safetensors(str(tmp), data)
            tmp.replace(path)
        index.append(
            {
                "id": w["id"],
                "group": w["group"] + "/" + w["region"],
                "opening": len(p),
                "positions": len(y),
                "prefix": fingerprint(p),
                "labels": fingerprint(y),
                "file": path.name,
                "sha256": digest(path),
            }
        )
        if (i + 1) % 200 == 0:
            print("features", i + 1, flush=True)
    return index


def logits(net, step, args, *, visits, differentiate):
    pre, h, k, v, pos, y, _ = args
    mask = "causal" if y.size > 1 else None
    z = net.refine(
        pre, h, k, v, pos, mask, step=step, visits=visits, differentiate=differentiate
    )
    return net.project(net.text.model.norm(z)).astype(mx.float32)


def logprob(z):
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def objective(net, close, eos, kl):
    def loss(step, t, *args):
        y = args[-2]
        p = logprob(logits(net, step, args, visits=2, differentiate=True))
        nll = -mx.take_along_axis(p, y[None, :, None], axis=-1)[0, :, 0]
        weights = mx.where((y == close) | (y == eos), 3.0, 1.0)
        anchor = (mx.exp(t) * (t - p)).sum() / y.size
        return (nll * weights).mean() + kl * anchor, nll.mean(), anchor

    return nn.value_and_grad(net.step, loss)


def probe(net, idx, folder, s, guard, visits):
    enrolled = set(s["probes"])
    out = {}
    for r in idx:
        if r["id"] not in enrolled:
            continue
        total = 0.0
        for args in parts(r, folder, {}, s):
            guard.check()
            p = logprob(logits(net, net.step, args, visits=visits, differentiate=False))
            total -= (
                mx.take_along_axis(p, args[-2][None, :, None], axis=-1).sum().item()
            )
        out[r["id"]] = total / r["positions"]
    assert set(out) == enrolled
    return out


def verdict(s, idx, history, probes):
    assert set(probes) == {"before", "after", "one"}
    groups = defaultdict(list)
    second = []
    for r in idx:
        k = r["id"]
        if k not in probes["before"]:
            continue
        groups[r["group"]].append(probes["before"][k] - probes["after"][k])
        second.append(probes["one"][k] - probes["after"][k])
    gains = {k: mean(v) for k, v in groups.items()}
    retain = mean(v for k, v in gains.items() if k.endswith("/retain"))
    gates = {
        "exposure": sum(r["positions"] for r in history) == s["expected_positions"],
        "finite": all(
            math.isfinite(r["gradient"])
            and r["gradient"] > 0
            and math.isfinite(r["loss"])
            for r in history
        ),
        "clean_reasoning": gains["strategy/clean"] > 0,
        "learner_reasoning": gains["strategy/learner"] > 0,
        "clean_action": gains["report_action/clean"] > 0,
        "learner_action": gains["report_action/learner"] > 0,
        "protected": retain >= -0.05,
        "second_read": mean(second) > 0.002,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "gains": gains,
        "protected_gain": retain,
        "second_read_gain": mean(second),
        "updates": len(history),
        "positions": sum(r["positions"] for r in history),
    }


def fit(name="learner-prefix-fit-01"):
    s = registered(name)
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/shared_query_fit.py",
            "loopqwen/research/learner_prefix_fit.py",
            "scripts/register_learner_prefix.py",
            "loopqwen/checkpoint.py",
        ),
    )
    guard = Budget(name, s["limits"])
    history = []
    try:
        actor = SharedQueryAssistant(s["checkpoint"], RESULTS / s["parent"])
        net = actor._tool_model
        teacher = QueryStep(net.args.hidden_size, s["rank"], s["gate_rank"])
        original = mx.load(str(RESULTS / s["parent"] / "step.safetensors"))
        teacher.load_weights(list(original.items()), strict=True)
        teacher.freeze()
        opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)
        folder = CACHE / name / digest(root / "registration.json")
        folder.mkdir(parents=True, exist_ok=True)
        idx = extract(net, unpack(Path(s["training"])), folder, guard)
        packed(root / "feature-index.json.gz", idx)
        before = probe(net, idx, folder, s, guard, 2)
        close = actor.tokenizer.convert_tokens_to_ids("</think>")
        eos = actor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        fn = objective(net, close, eos, s["teacher_kl"])
        steps = schedule(idx, s)
        for j, (epoch, ids) in enumerate(steps):
            total = sum(idx[i]["positions"] for i in ids)
            grad = None
            metrics = mx.zeros((3,), dtype=mx.float32)
            for i in ids:
                for args in parts(idx[i], folder, {}, s):
                    guard.check()
                    # Keep the original deployment kernel for the frozen target. MLX's
                    # single-query fused kernel does not expose an attention backward pass.
                    target_logp = logprob(
                        logits(net, teacher, args, visits=2, differentiate=False)
                    )
                    mx.eval(target_logp)
                    (obj, nll, anchor), g = fn(net.step, target_logp, *args)
                    scale = args[-2].size / total
                    grad = accumulate(grad, g, scale)
                    metrics = metrics + scale * mx.stack([obj, nll, anchor])
                    mx.eval(grad, metrics)
            norm = apply(net.step, opt, grad)
            loss, nll, anchor = metrics.tolist()
            if (
                not all(math.isfinite(v) for v in (norm, loss, nll, anchor))
                or norm <= 0
            ):
                raise InterruptedError("Nonfinite or zero optimizer update")
            history.append(
                {
                    "epoch": epoch,
                    "rows": [idx[i]["id"] for i in ids],
                    "positions": total,
                    "gradient": norm,
                    "loss": loss,
                    "nll": nll,
                    "parent_kl": anchor,
                }
            )
            if j + 1 == len(steps) or steps[j + 1][0] != epoch:
                checkpoint.write(
                    folder,
                    net.step,
                    opt,
                    {
                        "step": j + 1,
                        "epoch": epoch,
                        "history": history,
                        "registration": digest(root / "registration.json"),
                    },
                )
            if (j + 1) % 80 == 0:
                print("updates", j + 1, "of", len(steps), flush=True)
        packed(root / "training-log.json.gz", history)
        weights = root / "step.safetensors"
        learned = dict(tree_flatten(net.step.trainable_parameters()))
        mx.save_safetensors(str(weights), learned)
        save(
            root / "fit.json",
            {
                "weights": digest(weights),
                "registration": digest(root / "registration.json"),
                "backbone": read(RESULTS / s["parent"] / "fit.json")["backbone"],
                "updates": len(history),
                "trainable": sum(v.size for v in learned.values()),
            },
        )
        probes = {
            "before": before,
            "after": probe(net, idx, folder, s, guard, 2),
            "one": probe(net, idx, folder, s, guard, 1),
        }
        save(root / "probes.json", probes)
        decision = verdict(s, idx, history, probes)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        packed(root / "partial-training-log.json.gz", history)
        decision = {
            "passed": False,
            "status": "interrupted",
            "reason": str(e),
            "updates": len(history),
        }
    finally:
        guard.close()
    save(root / "decision.json", decision)
    print(decision, flush=True)


def verify(root, d):
    s = registered(read(root / "registration.json")["id"])
    assert s == read(root / "registration.json") and all(
        digest(Path(k)) == v for k, v in s["sources"].items()
    )
    if d.get("status") == "interrupted":
        assert (
            not d["passed"]
            and d["reason"]
            and len(unpack(root / "partial-training-log.json.gz")) == d["updates"]
        )
        return
    idx = unpack(root / "feature-index.json.gz")
    rows = unpack(Path(s["training"]))
    for r, w in zip(idx, rows, strict=True):
        assert (r["id"], r["prefix"], r["labels"], r["opening"], r["positions"]) == (
            w["id"],
            fingerprint(w["p"]),
            fingerprint(w["y"]),
            len(w["p"]),
            len(w["y"]),
        )
    history = unpack(root / "training-log.json.gz")
    for r, (epoch, positions) in zip(history, schedule(idx, s), strict=True):
        assert (
            r["epoch"] == epoch
            and r["rows"] == [idx[i]["id"] for i in positions]
            and r["positions"] == sum(idx[i]["positions"] for i in positions)
        )
    assert verdict(s, idx, history, read(root / "probes.json")) == d
    fit = read(root / "fit.json")
    assert (
        fit["trainable"] == 721920
        and fit["weights"] == digest(root / "step.safetensors")
        and fit["weights"] != read(RESULTS / s["parent"] / "fit.json")["weights"]
    )
    assert fit["backbone"] == read(RESULTS / s["parent"] / "fit.json")["backbone"]


if __name__ == "__main__":
    fit()
