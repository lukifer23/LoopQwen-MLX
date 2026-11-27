"""Fit only a function of true recurrent changes on native, frozen response-position states."""

import math
import random
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from .. import checkpoint
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..model_loader import count
from .native_function_data import token_pair
from .visit_residual_probe import read as interchange


def annotations(tok, view, y):
    close = tok.convert_tokens_to_ids("</think>")
    body = y.index(close) + 1
    visible = next((i for i in range(body, len(y)) if tok.decode([y[i]]).strip()), body)
    bad = None
    if view["alternative"]:
        _, alt = token_pair(tok, {**view, "target": view["alternative"]})
        bad = next(((j, t) for j, (s, t) in enumerate(zip(y, alt)) if s != t), None)
        # Old source alternatives can have a different rationale; only contrast emitted visible tokens.
        if bad and bad[0] < body:
            bad = None
    return body, visible, bad


def stream(views, seed):
    rng = random.Random(seed)
    pools = defaultdict(list)
    pairs = defaultdict(list)
    for i, v in enumerate(views):
        pools[v["group"]].append(i)
        if v["group"] == "effects":
            pairs[v["source"]].append(i)
    natural = list(pairs.values())
    assert all(len(v) == 2 for v in natural)
    tools = pools["dependency"] + pools["same_version"]
    preserved = (
        "native-function",
        "report-action",
        "cross-view",
        "proof-curriculum",
        "followup-protected",
    )
    while True:
        pair = rng.choice(natural)
        old = rng.sample(preserved, 3)
        yield [
            *pair,
            rng.choice(tools),
            rng.choice(tools),
            *[rng.choice(pools[g]) for g in old],
        ]


def packed_rows(indices, bank, seed, step, limit):
    rng = random.Random(f"{seed}/{step}")
    one = []
    two = []
    target = []
    body = []
    anchors = []
    differences = []
    foils = []
    positions = []
    for slot, i in enumerate(indices):
        path, labels, start, visible, bad = bank[i]
        mandatory = {0, visible, len(labels) - 1}
        if bad:
            mandatory.add(bad[0])
        remaining = [j for j in range(len(labels)) if j not in mandatory]
        selected = sorted(
            mandatory
            | set(rng.sample(remaining, min(len(remaining), limit - len(mandatory))))
        )
        positions.append(selected)
        if bad:
            differences.append(len(target) + selected.index(bad[0]))
            foils.append(bad[1])
        states = mx.load(str(path))
        at = mx.array(selected)
        one.append(states["first"][at])
        two.append(states["second"][at])
        anchors.extend([slot >= 4] * len(selected))
        body.extend(j >= start for j in selected)
        target.extend(labels[j] for j in selected)
    tensors = (
        mx.concatenate(one),
        mx.concatenate(two),
        mx.array(target),
        mx.array(body),
        mx.array(anchors),
        mx.array(differences, dtype=mx.int32),
        mx.array(foils, dtype=mx.int32),
    )
    return tensors, positions


def gradient_function(net):
    def loss(head, first, second, targets, visible, anchor, js, foils):
        def project(x):
            return net.project(net.text.model.norm(x)).astype(mx.float32)

        logits = project(head(first, second))
        ce = nn.losses.cross_entropy(logits, targets)
        # Keep the reference distribution over the full vocabulary on every sampled rehearsal token.
        base = mx.stop_gradient(project(second))
        log_base = base - mx.logsumexp(base, axis=-1, keepdims=True)
        log_candidate = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        kl = mx.sum(mx.exp(log_base) * (log_base - log_candidate), axis=-1)
        anchored = (kl * anchor).sum() / anchor.sum()
        active = 1 - anchor.astype(mx.float32)
        semantic = active * visible
        primary = 0.35 * (ce * active).sum() / active.sum() + 0.65 * (
            ce * semantic
        ).sum() / mx.maximum(semantic.sum(), 1)
        memory = (ce * anchor).sum() / anchor.sum()
        margin = logits[js, targets[js]] - logits[js, foils]
        ranking = mx.logaddexp(mx.zeros_like(margin), 1.0 - margin).mean()
        total = 0.45 * primary + 0.15 * ranking + 0.20 * memory + 0.20 * anchored
        return total, primary, ranking, memory, anchored

    return nn.value_and_grad(net.refinement, loss)


def fit(net, tok, s, root, guard):
    start = perf_counter()
    examples = read(s["training"])["examples"]
    weights = root / "visit-residual.safetensors"
    with TemporaryDirectory(dir=root.parent) as tmp:
        folder = Path(tmp)
        bank = []
        proof = []
        baseline = LoopedModel(net.text, net.span, alpha=0.35)
        baseline.alpha_logit = net.alpha_logit
        baseline.alpha_override = net.alpha_override
        baseline.eval()
        for i, v in enumerate(examples):
            guard.check()
            prefix, y = token_pair(tok, v)
            ids = mx.array([prefix + y[:-1]])
            first, second = net.visits(ids)
            start_at = len(prefix) - 1
            if i == 0:
                native = baseline(ids)[:, -1]
                patched = net(ids)[:, -1]
                exact = mx.array_equal(native, patched).item()
                assert exact and native.shape[-1] == 248320
                save(
                    root / "native-parity.json",
                    {
                        "view": v["id"],
                        "input": fingerprint(ids.tolist()[0]),
                        "all_logits_exact": exact,
                        "vocabulary": native.shape[-1],
                        "physical_sites": [list(x) for x in net.sites],
                        "different_visits": bool(mx.any(first != second).item()),
                    },
                )
                assert read(root / "native-parity.json")["different_visits"]
            path = folder / (str(i) + ".safetensors")
            mx.save_safetensors(
                str(path),
                {"first": first[0, start_at:], "second": second[0, start_at:]},
            )
            body, visible, bad = annotations(tok, v, y)
            bank.append((path, y, body, visible, bad))
            proof.append(
                {
                    "id": v["id"],
                    "prompt": fingerprint(prefix),
                    "targets": len(y),
                    "target_ids_sha256": fingerprint(y),
                    "values": digest(path),
                    "body": body,
                    "first_visible": visible,
                    "contrast": bad,
                }
            )
            if (i + 1) % 400 == 0:
                print("Cached actual recurrent training views", i + 1, flush=True)
        packed(root / "training-features.json.gz", proof)
        prepared = perf_counter() - start
        opt = optim.AdamW(s["lr"], weight_decay=0.01)
        grad = gradient_function(net)
        draw = stream(examples, s["fit_seed"])
        checkpoints = Path("checkpoints/research") / s["id"]
        checkpoints.mkdir(exist_ok=True, parents=True)
        binding = {
            "training": digest(Path(s["training"])),
            "spec": digest(root / "registration.json"),
            "features": digest(root / "training-features.json.gz"),
        }
        state = {
            "step": 0,
            "training_identity": binding,
            "rows": [],
            "sampled_targets": 0,
            "training_seconds": 0.0,
        }
        if (checkpoints / "latest.json").is_file():
            state = checkpoint.restore(checkpoints, net.refinement, opt)
            assert (
                state["training_identity"] == binding
                and state["step"] == opt.state["step"].item()
            )
            for _ in range(state["step"]):
                next(draw)
        for step in range(state["step"] + 1, s["updates"] + 1):
            guard.check()
            tick = perf_counter()
            batch = next(draw)
            data, positions = packed_rows(
                batch, bank, s["fit_seed"], step, s["positions_per_view"]
            )
            assert (
                len(data[-1]) >= 2
            )  # Both genuinely contrasting questions have a different answer.
            losses, g = grad(net.refinement, *data)
            g, grad_norm = optim.clip_grad_norm(g, s["max_gradient_norm"])
            mx.eval(losses, g, grad_norm)
            vals = [l.item() for l in losses]
            if not all(math.isfinite(v) for v in [*vals, grad_norm.item()]):
                raise InterruptedError("Nonfinite residual objective")
            opt.update(net.refinement, g)
            mx.eval(net.refinement.parameters(), opt.state)
            state["rows"].append(
                {
                    "step": step,
                    "views": batch,
                    "sampled": len(data[2]),
                    "positions": positions,
                    "contrasts": len(data[-1]),
                    "losses": vals,
                    "gradient_norm": grad_norm.item(),
                }
            )
            state["sampled_targets"] += len(data[2])
            state["training_seconds"] += perf_counter() - tick
            if step % 100 == 0:
                state["step"] = step
                checkpoint.write(checkpoints, net.refinement, opt, state)
                print("Residual step", step, [round(v, 4) for v in vals], flush=True)
        # The learned correction must still be exactly zero when the two visits are made identical.
        probe = mx.load(str(bank[0][0]))["first"]
        assert mx.array_equal(net.refinement(probe, probe), probe).item()
        assert mx.any(net.refinement.output.weight != 0).item()
        guard.check()
        packed(root / "training-interchange.json.gz", interchange(net, bank, examples))
    mx.save_safetensors(str(weights), dict(tree_flatten(net.refinement.parameters())))
    packed(root / "updates.json.gz", state["rows"])
    save(
        root / "fit.json",
        {
            "weights": digest(weights),
            "passes": 2,
            "trainable": count(net, True),
            "updates": state["step"],
            "views": len(examples),
            "targets": state["sampled_targets"],
            "training": binding["training"],
            "feature_sha256": binding["features"],
            "extraction_seconds": prepared,
            "optimizer_seconds": state["training_seconds"],
            "peak": mx.get_peak_memory(),
            "zero_without_change": True,
        },
    )
