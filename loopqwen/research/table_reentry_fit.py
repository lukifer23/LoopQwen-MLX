"""Fit a Base-only four-layer recurrence, checking that its original first visit stays fixed."""

import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..kernels.gdn_backend import select
from ..looped_model import LoopedModel
from ..recurrence import Span
from ..reentry import from_base, packed_path, resident
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .integrated_assistance_fit import positions
from .native_function_data import token_pair
from .registered import registered

NAME = "table-reentry-fit-01"
CODES = (
    "loopqwen/reentry.py",
    "loopqwen/research/table_reentry_fit.py",
    "loopqwen/research/table_worlds.py",
    "loopqwen/observation_review.py",
    "loopqwen/looped_model.py",
    "loopqwen/kernels/gdn_backend.py",
    "loopqwen/recurrent_cache.py",
    "loopqwen/research/integrated_assistance_fit.py",
    "loopqwen/research/native_function_data.py",
    "scripts/table_policy_material.py",
    "pyproject.toml",
    "uv.lock",
)


def root_for(s):
    root = Path("outputs/research") / s["id"]
    root.mkdir(exist_ok=True, parents=True)
    if (root / "decision.json").exists():
        raise FileExistsError("The registered stage is already closed")
    if (root / "registration.json").exists():
        assert read(root / "registration.json") == s
    else:
        save(root / "registration.json", s)
    for path, sha in s["sources"].items():
        assert digest(Path(path)) == sha
    return root


def deck(w):
    family = w["group"]
    if family.startswith("table_"):
        return "changed"
    if family.startswith("database_"):
        return "database"
    return "language" if family in ("open", "reasoning") else "earlier"


def inputs(tok, s):
    rows = read(s["training"])["examples"]
    items = {}
    close = tok.convert_tokens_to_ids("</think>")
    for w in rows:
        prefix, labels = token_pair(tok, w)
        if len(prefix) + len(labels) > s["warm_train_limit"]:
            continue
        visible = labels.index(close) + 1
        while visible < len(labels) - 1 and not tok.decode([labels[visible]]).strip():
            visible += 1
        items[w["id"]] = (deck(w), prefix, labels, visible)
    return items


def decks(items, seed):
    groups = defaultdict(list)
    for key, (group, *_) in items.items():
        groups[group].append(key)

    def shuffled(group):
        rng = random.Random(f"{seed}/{group}")
        while True:
            batch = sorted(groups[group])
            rng.shuffle(batch)
            yield from batch

    return {g: shuffled(g) for g in sorted(groups)}


def optimizer(lr, s):
    return optim.MultiOptimizer(
        [
            optim.AdamW(lr * s["gate_multiplier"], weight_decay=0.01),
            optim.AdamW(lr, weight_decay=0.01),
        ],
        [lambda name, _: name == "mixing"],
    )


def gold_nll(net, h, ix, labels):
    hidden = net.second_visit(h)
    return nll_from(net, hidden, ix, labels).mean()


def nll_from(net, h, ix, labels):
    logits = net.project(net.text.model.norm(h[0, ix])).astype(mx.float32)
    return nn.losses.cross_entropy(logits, labels, reduction="none")


def accumulate(total, g, scale):
    if total is None:
        return tree_map(lambda x: x.astype(mx.float32) * scale, g)
    return tree_map(lambda a, x: a + x.astype(mx.float32) * scale, total, g)


def apply(net, opt, gradient):
    gradient, norm = optim.clip_grad_norm(gradient, 1.0)
    mx.eval(gradient, norm)
    if not math.isfinite(norm.item()):
        raise InterruptedError("Nonfinite native gradient")
    opt.update(net, gradient)
    mx.eval(net.trainable_parameters(), opt.state)
    return norm.item()


def parity(net, ids):
    net.eval()
    seq = mx.array([ids])
    original = LoopedModel(net.text, Span(net.span.start, 4, 1))
    original.eval()
    dense = original.project(original.hidden(seq)[:, -1])
    observed = net.project(net.hidden(seq)[:, -1])
    a = net.first_visit(seq)
    cache = net.make_cache()
    # Two prefills with independent native states at each physical site.
    mid = len(ids) // 2
    net.hidden(seq[:, :mid], cache)
    incremental = net.project(net.hidden(seq[:, mid:], cache)[:, -1])
    diff = mx.max(
        mx.abs(incremental.astype(mx.float32) - observed.astype(mx.float32))
    ).item()
    return {
        "exact_ordinary_logits": mx.array_equal(dense, observed).item(),
        "vocabulary": dense.shape[-1],
        "ids": ids,
        "first_state": fingerprint(a[0, -1].astype(mx.float32).tolist()),
        "sites": len(cache.slots),
        "unique_slots": len(set(map(id, cache.slots.values()))),
        "gdn_state_differences": [
            [
                mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
                for a, b in zip(
                    cache.slots[i, 0].state, cache.slots[i, 1].state, strict=True
                )
            ]
            for i in range(net.span.start, net.span.start + 3)
        ],
        "attention_offsets": [
            cache.slots[len(net.text.layers) - 1, p].offset for p in (0, 1)
        ],
        "full_top": mx.argmax(observed).item(),
        "incremental_top": mx.argmax(incremental).item(),
        "max_logit_error": diff,
    }


def portable(net, s, root, opening):
    path = packed_path(s["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        output = Path(tmp) / path.name
        mx.save_safetensors(
            str(output), dict(sorted(tree_flatten(net.text.parameters())))
        )
        sha = digest(output)
        if path.exists():
            if digest(path) != sha:
                raise ValueError(
                    "Existing prequantized Base differs from direct native packing"
                )
        else:
            output.replace(path)
    identity = {
        "sha256": sha,
        "bytes": path.stat().st_size,
        "base": s["base_text_sha256"],
        "config": s["config_sha256"],
    }
    other = resident(s["checkpoint"], root, s, identity)
    net.eval()
    other.eval()
    seq = mx.array([opening])
    a = net.project(net.hidden(seq)[:, -1])
    b = other.project(other.hidden(seq)[:, -1])
    exact = mx.array_equal(a, b).item()
    save(
        root / "packed-base.json",
        {
            **identity,
            "exact_logits": exact,
            "compared": a.size,
            "input": fingerprint(opening),
        },
    )
    if not exact:
        raise InterruptedError("Prepacked Base failed full-vocabulary reload parity")
    return identity


def train(net, tok, s, root, guard):
    items = inputs(tok, s)
    seed = s["fit_seed"]
    chosen = min(items, key=lambda k: fingerprint([seed, "parity", k]))
    check = parity(net, items[chosen][1])
    save(root / "initial-parity.json", check)
    if not (
        check["sites"] == check["unique_slots"] == 28
        and check["vocabulary"] == 248320
        and check["exact_ordinary_logits"]
        and check["full_top"] == check["incremental_top"]
    ):
        raise InterruptedError(
            "Initial ordinary logits or recurrent cache failed the registered parity check"
        )
    pack_identity = portable(net, s, root, items[chosen][1])
    streams = decks(items, seed)
    opt = optimizer(s["warm_lr"], s)
    directory = Path("checkpoints/research") / NAME
    directory.mkdir(exist_ok=True, parents=True)
    identity = digest(root / "registration.json")
    state = {"registration": identity, "step": 0, "log": []}
    if (directory / "latest.json").exists():
        state = checkpoint.restore(directory, net, opt)
        assert state["registration"] == identity
        for _ in range(state["step"]):
            for stream in streams.values():
                next(stream)
    select(net, "native")
    grad = nn.value_and_grad(net, gold_nll)
    with TemporaryDirectory(dir=root.parent) as tmp:
        paths = {}
        for idx, (key, (_, prefix, labels, _)) in enumerate(items.items()):
            guard.check()
            path = Path(tmp) / f"{idx}.safetensors"
            h = net.first_visit(mx.array([prefix + labels[:-1]]))[0]
            mx.save_safetensors(str(path), {"hidden": h})
            paths[key] = path
        net.train()
        net.checkpointing = True
        net.checkpoint_policy = "span"
        for step in range(state["step"] + 1, s["warm_updates"] + 1):
            guard.check()
            gtotal = None
            observations = []
            for stream in streams.values():
                key = next(stream)
                group, prefix, y, visible = items[key]
                selected = positions(
                    y, visible, s["warm_targets"], f"{seed}/{step}/{key}"
                )
                h = mx.load(str(paths[key]))["hidden"][None]
                loss, g = grad(
                    net,
                    h,
                    mx.array(selected) + len(prefix) - 1,
                    mx.array([y[j] for j in selected]),
                )
                gtotal = accumulate(gtotal, g, 1 / len(streams))
                mx.eval(loss, gtotal)
                if not math.isfinite(loss.item()):
                    raise InterruptedError("Nonfinite native cross entropy")
                observations.append(
                    {"view": key, "positions": selected, "loss": loss.item()}
                )
            norm = apply(net, opt, gtotal)
            state["step"] = step
            state["log"].append(
                {"step": step, "observations": observations, "gradient_norm": norm}
            )
            if step % 64 == 0 or step == s["warm_updates"]:
                checkpoint.write(directory, net, opt, state)
                print(
                    "Base four-layer warm-up",
                    step,
                    round(mean(x["loss"] for x in observations), 4),
                    flush=True,
                )
        net.eval()
        probes = []
        for group in streams:
            pool = [k for k, v in items.items() if v[0] == group]
            for key in sorted(pool, key=lambda k: fingerprint([seed, "probe", k]))[:8]:
                _, prefix, y, visible = items[key]
                js = positions(y, visible, s["warm_targets"], f"{seed}/probe/{key}")
                ix = mx.array(js) + len(prefix) - 1
                labels = mx.array([y[j] for j in js])
                first = mx.load(str(paths[key]))["hidden"][None]
                second = net.second_visit(first)
                probes.append(
                    {
                        "view": key,
                        "group": group,
                        "positions": js,
                        "first_nll": nll_from(net, first, ix, labels).mean().item(),
                        "second_nll": nll_from(net, second, ix, labels).mean().item(),
                    }
                )
    weights = root / "reentry.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(net.trainable_parameters())))
    shutil.copyfile(
        checkpoint.location(directory) / "optimizer.safetensors",
        root / "optimizer.safetensors",
    )
    packed(root / "optimizer-steps.json.gz", state["log"])
    after = parity(net, items[chosen][1])
    save(root / "trained-parity.json", after)
    save(root / "training-probes.json", probes)
    stats = {
        "weights": digest(weights),
        "registration": identity,
        "views": len(items),
        "backbone": pack_identity,
        "updates": state["step"],
        "gate_rms": mx.sqrt(mx.mean(mx.square(mx.tanh(net.mixing)))).item(),
        "residual_norms": [mx.linalg.norm(r.up.weight).item() for r in net.residuals],
        "trainable": sum(v.size for _, v in tree_flatten(net.trainable_parameters())),
    }
    save(root / "fit.json", stats)
    return summarize(check, after, stats, probes)


def summarize(before, after, fit, probes):
    gain = mean(r["first_nll"] - r["second_nll"] for r in probes)
    checks = {
        "exact_initial": before["exact_ordinary_logits"],
        "independent_histories": before["sites"] == before["unique_slots"] == 28
        and all(
            len(row) == 2 and min(row) > 0 for row in after["gdn_state_differences"]
        ),
        "ordinary_unchanged": before["first_state"] == after["first_state"],
        "cached_top": before["full_top"] == before["incremental_top"]
        and after["full_top"] == after["incremental_top"],
        "trained_change": fit["gate_rms"] > 0
        and all(v > 0 and math.isfinite(v) for v in fit["residual_norms"]),
        "probe_gain": gain >= 0.05,
    }
    return {
        "passed": all(checks.values()),
        "gates": checks,
        "training_mean_nll_gain": gain,
        "training_precondition_only": True,
    }


def verify(root, d):
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    s = registered(NAME)
    assert s == read(root / "registration.json")
    items = inputs(tokenizer(s), s)
    streams = decks(items, s["fit_seed"])
    fit = read(root / "fit.json")
    log = unpack(root / "optimizer-steps.json.gz")
    assert (
        len(log) == fit["updates"] == s["warm_updates"]
        and fit["trainable"] == 1049600
        and fit["views"] == len(items)
    )
    assert (
        digest(root / "reentry.safetensors") == fit["weights"]
        and digest(root / "registration.json") == fit["registration"]
    )
    for n, row in enumerate(log, 1):
        assert n == row["step"] and math.isfinite(row["gradient_norm"])
        for stream, r in zip(streams.values(), row["observations"], strict=True):
            key = next(stream)
            _, _, y, v = items[key]
            assert r["view"] == key and math.isfinite(r["loss"])
            assert r["positions"] == positions(
                y, v, s["warm_targets"], f"{s['fit_seed']}/{n}/{key}"
            )
    assert (
        summarize(
            read(root / "initial-parity.json"),
            read(root / "trained-parity.json"),
            fit,
            read(root / "training-probes.json"),
        )
        == d
    )


def run():
    s = registered(NAME)
    prev = Path("research/results/integrated-assistance-01/decision.json")
    old = read(prev)
    reading = read("docs/measurements/integrated-assistance-reading.json")
    if "arms" not in old or (old["passed"] and reading["reading_gate"]):
        raise ValueError("Resolve the preceding scored primary first")
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    root = root_for(s)
    capture(root, CODES)
    guard = Budget(NAME, s["limits"])
    try:
        d = train(from_base(s["checkpoint"], s), tokenizer(s), s, root, guard)
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        d = {"passed": False, "status": "interrupted", "reason": str(exc)}
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
