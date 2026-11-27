"""Differentiate through tied, causal reads of actual source memory on Apple Silicon."""

import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx import nn
from mlx import optimizers as optim
from mlx.utils import tree_flatten

from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..shared_query_assistant import from_backbone
from .conditional_features import feature_root
from .core import Budget
from .file_evidence_environment import capture
from .registered import registered
from .shared_query_features import FEATURES, FIT, OLD, RESULTS, SOURCES
from .table_reentry_fit import accumulate, apply, root_for


def schedule(idx, s):
    rng = random.Random(s["fit_seed"])
    out = []
    for epoch in range(s["epochs"]):
        seq = list(range(len(idx)))
        rng.shuffle(seq)
        batch = []
        total = 0
        for i in seq:
            batch.append(i)
            total += idx[i]["positions"]
            if total >= s["update_positions"]:
                out.append((epoch, batch))
                batch = []
                total = 0
        if batch:
            out.append((epoch, batch))
    return out


def cited_loss(query, keys, pos, annotations, scale):
    """Negative log probability assigned to the cited input, averaged over query heads."""
    total = mx.array(0.0)
    for local, ranges in annotations:
        end = pos + local + 1
        k = keys[:, :, :end].astype(mx.float32)
        batch, heads, _, dim = query.shape
        q = (
            query[:, :, local : local + 1]
            .astype(mx.float32)
            .reshape(batch, k.shape[1], heads // k.shape[1], 1, dim)
        )
        score = ((q * scale) @ k[:, :, None].swapaxes(-2, -1)).reshape(
            batch, heads, end
        )
        ix = mx.arange(end)
        mask = mx.zeros((end,), dtype=mx.bool_)
        for first, last in ranges:
            mask = mask | ((first <= ix) & (ix < last))
        valid = mx.where(mask, score, -mx.inf)
        logmass = mx.logsumexp(valid, axis=-1) - mx.logsumexp(score, axis=-1)
        total = total + math.log(heads) - mx.logsumexp(logmass, axis=-1).mean()
    return total


def loss_fn(net, s, *, differentiate, first_only=False):
    def loss(step, pre, h, k, v, pos, labels, annotations):
        queries = []
        mask = "causal" if h.shape[1] > 1 else None
        z = net.refine(
            pre,
            h,
            k,
            v,
            pos,
            mask,
            step=step,
            differentiate=differentiate,
            observe_query=queries.append,
            visits=1 if first_only else None,
        )
        logits = net.project(net.text.model.norm(z)).astype(mx.float32)
        teacher = mx.stop_gradient(
            net.project(net.text.model.norm(h)).astype(mx.float32)
        )
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logt = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(logp, labels[None, :, None], axis=-1).sum()
        baseline = -mx.take_along_axis(logt, labels[None, :, None], axis=-1).sum()
        kl = (mx.exp(logt) * (logt - logp)).sum()
        assert len(queries) == 1
        evidence = cited_loss(
            queries[0], k, pos, annotations, net.text.layers[-1].self_attn.scale
        )
        objective = (
            ce + s["teacher_kl"] * kl + s["evidence_weight"] * evidence
        ) / labels.size
        return objective, ce, baseline, kl, evidence

    return loss


def parts(item, folder, evidence, s):
    d = mx.load(str(folder / item["file"]))
    count = item["positions"]
    first = item["opening"] - 1
    annotation = evidence.get(item["id"])
    if annotation:
        assert (
            annotation["prefix"] == item["prefix"]
            and annotation["labels"] == item["labels"]
        )
    for start in range(0, count, s["micro_positions"]):
        end = min(count, start + s["micro_positions"])
        marks = [
            (r["query"] - start, r["source"])
            for r in (annotation["evidence"] if annotation else [])
            if start <= r["query"] < end
        ]
        yield (
            d["pre"][:, start:end],
            d["ordinary"][:, start:end],
            d["keys"][:, :, : first + end],
            d["values"][:, :, : first + end],
            first + start,
            d["labels"][start:end],
            marks,
        )


def probe(net, idx, folder, evidence, s, guard=None, first_only=False):
    fn = loss_fn(net, s, differentiate=False, first_only=first_only)
    groups = defaultdict(list)
    for i, r in enumerate(idx):
        groups[r["group"]].append(i)
    out = []
    for group, ids in groups.items():
        sample = sorted(
            ids, key=lambda i: fingerprint([s["fit_seed"], "probe", idx[i]["id"]])
        )[:4]
        for i in sample:
            item = idx[i]
            tot = [0.0] * 4
            n = 0
            for args in parts(item, folder, evidence, s):
                if guard is not None:
                    guard.check()
                vals = fn(net.step, *args)[1:]
                mx.eval(vals)
                tot = [a + b.item() for a, b in zip(tot, vals, strict=True)]
                n += len(args[-1])
            ce, base, kl, ev = tot
            out.append(
                {
                    "id": item["id"],
                    "group": group,
                    "positions": item["positions"],
                    "ce": ce / item["positions"],
                    "ordinary_ce": base / item["positions"],
                    "kl": kl / item["positions"],
                    "citations": n,
                    "citation_nll": ev,
                }
            )
    return out


def optimize(net, s, idx, folder, evidence, root, guard):
    fn = nn.value_and_grad(net.step, loss_fn(net, s, differentiate=True))
    opt = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    history = []
    for j, (epoch, ids) in enumerate(schedule(idx, s), 1):
        total = sum(idx[i]["positions"] for i in ids)
        grad = None
        cost = [mx.array(0.0) for _ in range(4)]
        observed = 0
        citations = 0
        for i in ids:
            for args in parts(idx[i], folder, evidence, s):
                guard.check()
                (obj, ce, base, kl, ev), g = fn(net.step, *args)
                size = args[-2].size
                grad = accumulate(grad, g, size / total)
                cost = [a + b for a, b in zip(cost, (ce, base, kl, ev), strict=True)]
                mx.eval(grad, cost, obj)
                observed += size
                citations += len(args[-1])
        assert total == observed
        norm = apply(net.step, opt, grad)
        if norm <= 0 or not all(math.isfinite(a.item()) for a in cost):
            raise InterruptedError("Nonfinite loss or zero gradient")
        history.append(
            {
                "update": j,
                "epoch": epoch,
                "rows": [idx[i]["id"] for i in ids],
                "positions": total,
                "citations": citations,
                "ce": cost[0].item(),
                "ordinary_ce": cost[1].item(),
                "kl": cost[2].item(),
                "citation_nll": cost[3].item(),
                "gradient": norm,
            }
        )
        if j % 50 == 0:
            packed(root / "partial-log.json.gz", history)
            print("Native tied-query update", j, "epoch", epoch + 1, flush=True)
    mx.save_safetensors(
        str(root / "step.safetensors"), dict(tree_flatten(net.step.parameters()))
    )
    mx.save_safetensors(
        str(root / "optimizer.safetensors"), dict(tree_flatten(opt.state))
    )
    return history


def decision(s, history, before, after, one):
    assert (
        [r["id"] for r in before] == [r["id"] for r in after] == [r["id"] for r in one]
    )
    by = defaultdict(list)
    depth = defaultdict(list)
    for a, b, c in zip(before, after, one, strict=True):
        by[a["group"]].append(a["ce"] - b["ce"])
        depth[a["group"]].append(c["ce"] - b["ce"])
    groups = {k: mean(z) for k, z in by.items()}
    second = mean(mean(z) for z in depth.values())
    numerator = sum(
        a["citation_nll"] - b["citation_nll"]
        for a, b in zip(before, after, strict=True)
    )
    citations = sum(r["citations"] for r in before)
    gates = {
        "all_tokens": sum(x["positions"] for x in history) == s["expected_positions"],
        "all_citations": sum(x["citations"] for x in history)
        == s["expected_citations"],
        "finite_updates": all(
            math.isfinite(x["gradient"]) and x["gradient"] > 0 for x in history
        ),
        "source_fit": groups["strategy"] > 0
        and groups["report_action"] > 0
        and mean(groups.values()) >= 0.04,
        "repeated_read": second > 0.002,
        "source_attention": citations > 0 and numerator / citations > 0.02,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": groups,
        "mean_second_visit_gain": second,
        "mean_citation_nll_gain": numerator / citations,
        "updates": len(history),
        "positions": sum(r["positions"] for r in history),
        "citations": sum(r["citations"] for r in history),
    }


def verify(root, d):
    s = registered(FIT)
    assert s == read(root / "registration.json")
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    archive = RESULTS / FEATURES
    idx = unpack(archive / "feature-index.json.gz")
    history = unpack(root / "training-log.json.gz")
    for r, (epoch, indices) in zip(history, schedule(idx, s), strict=True):
        assert r["epoch"] == epoch and r["rows"] == [idx[i]["id"] for i in indices]
        assert r["positions"] == sum(idx[i]["positions"] for i in indices)
    report = read(root / "probes.json")
    assert (
        decision(s, history, report["before"], report["after"], report["first_visit"])
        == d
    )
    fit = read(root / "fit.json")
    assert fit["weights"] == digest(root / "step.safetensors") and fit[
        "registration"
    ] == digest(root / "registration.json")
    assert fit["backbone"] == read(OLD / "backbone.json") and fit[
        "feature_archive"
    ] == digest(archive / "checksums.json")


def run():
    s = registered(FIT)
    archive = RESULTS / FEATURES
    if not read(archive / "decision.json")["passed"]:
        raise ValueError("Native memory features did not qualify")
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/shared_query_fit.py",
            "scripts/shared_query_evidence.py",
        ),
    )
    guard = Budget(FIT, s["limits"])
    try:
        fs = registered(FEATURES)
        folder = feature_root(fs)
        idx = unpack(archive / "feature-index.json.gz")
        evidence = unpack(Path(s["evidence"]))
        for r in idx:
            guard.check()
            assert digest(folder / r["file"]) == r["sha256"]
        ident = read(OLD / "backbone.json")
        net = from_backbone(s["checkpoint"], root, s, ident)
        before = probe(net, idx, folder, evidence, s, guard)
        history = optimize(net, s, idx, folder, evidence, root, guard)
        packed(root / "training-log.json.gz", history)
        (root / "partial-log.json.gz").unlink(missing_ok=True)
        after = probe(net, idx, folder, evidence, s, guard)
        one = probe(net, idx, folder, evidence, s, guard, True)
        save(
            root / "probes.json", {"before": before, "after": after, "first_visit": one}
        )
        save(
            root / "fit.json",
            {
                "weights": digest(root / "step.safetensors"),
                "registration": digest(root / "registration.json"),
                "backbone": ident,
                "feature_archive": digest(archive / "checksums.json"),
                "updates": len(history),
                "trainable": sum(
                    v.size for _, v in tree_flatten(net.step.parameters())
                ),
            },
        )
        d = decision(s, history, before, after, one)
    except (InterruptedError, TimeoutError, MemoryError) as e:
        d = {"passed": False, "status": "interrupted", "reason": str(e)}
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
