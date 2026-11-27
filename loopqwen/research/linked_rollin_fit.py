"""Fit the two physical visits on real action histories generated at two training checkpoints."""

import math
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint, qwen_tools
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..model_loader import count
from ..tool_model import Assistant
from .cross_view_study import tokenizer, write_documents
from .file_evidence_fit import pack
from .linked_rollin_data import (
    canonical,
    corrections,
    encode,
    protected_views,
    providers,
    score,
    selected_cases,
)
from .native_function_study import evaluate, replay


def demonstrations(training, folder):
    for w in training["rows"]:
        defs = qwen_tools.declarations(providers(w, folder))
        yield from canonical(w, defs)
    yield from protected_views()


def experiences(training, root, tok, folder, s, phase, *, check):
    selected = selected_cases(training, phase)
    archive = unpack(root / f"student-round-{phase}.json.gz")
    assert len(archive) == len(selected)
    views = []
    for w, r in zip(selected, archive, strict=True):
        assert r["id"] == w["id"]
        tools = providers(w, folder)
        q = r["requests"][0]
        if check:
            replay(
                tok,
                w,
                tools,
                [{"role": "user", "content": w["questions"][0]}],
                q,
                {**s, "max_tokens": s["rollin_tokens"]},
                "candidate",
            )
            assert q["score"] == score(tok, w, 0, q["trajectory"])
        views.extend(
            corrections(w, q["trajectory"], tok, qwen_tools.declarations(tools), phase)
        )
    return views


def turns(s, root):
    t = read(s["training"])
    tok = tokenizer(s)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_documents(t, folder)
        views = list(demonstrations(t, folder))
        for phase in range(2):
            views.extend(experiences(t, root, tok, folder, s, phase, check=True))
        return views


def loss_fn(model, h, rows, cols, y, semantic, j, bad):
    first = model._execute(h, model.sites[model.span.start :][:1])
    second = model._execute(first, model.sites[model.span.start :][1:])
    project = lambda h: model.project(model.text.model.norm(h)).astype(mx.float32)
    logits = project(second[rows, cols])
    ce = nn.losses.cross_entropy(logits, y)
    loss = 0.4 * ce.mean() + 0.6 * ce[semantic].mean()
    if j >= 0:
        prior = mx.stop_gradient(project(first[rows[j : j + 1], cols[j : j + 1]])[0])
        delta = logits[j, y[j]] - logits[j, bad]
        before = prior[y[j]] - prior[bad]
        absolute = mx.logaddexp(mx.array(0.0), 1.0 - delta)
        refinement = mx.logaddexp(mx.array(0.0), 0.35 + before - delta)
        loss = 0.4 * loss + 0.3 * ce[j] + 0.15 * absolute + 0.15 * refinement
    return loss


def collect(net, tok, s, t, folder, root, phase, guard):
    path = root / f"student-round-{phase}.json.gz"
    if path.exists():
        return
    net.eval()
    net.checkpointing = False
    actor = Assistant(s["checkpoint"], tool_format="qwen")
    actor.native_plain = True
    actor.tokenizer = tok
    actor._tool_model = net
    actor.reuse_prefix = True
    rows = selected_cases(t, phase)
    output = evaluate(
        {**s, "max_tokens": s["rollin_tokens"]},
        {"rows": rows},
        "learner",
        guard,
        lambda: actor,
        lambda w: providers(w, folder),
        scorer=score,
    )
    packed(path, output)


def fit(net, tok, s, root, guard, *, prepare):
    t = read(s["training"])
    folder = Path("checkpoints/research") / s["id"]
    folder.mkdir(parents=True, exist_ok=True)
    adam = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    grad = nn.value_and_grad(net, loss_fn)
    bind = digest(root / "registration.json")
    state = {
        "step": 0,
        "tokens": 0,
        "proof_tokens": 0,
        "contrast": 0,
        "loss": [],
        "seconds": 0,
        "draws": {},
        "registration": bind,
    }
    if (folder / "latest.json").exists():
        state = checkpoint.restore(folder, net, adam)
        assert (
            state["registration"] == bind and adam.state["step"].item() == state["step"]
        )
    draws = Counter(state["draws"])
    prior_time = state["seconds"]
    start = perf_counter()
    prepare_time = roll_time = 0.0
    phase_stats = []
    peak = mx.get_peak_memory()
    with TemporaryDirectory(dir=root.parent) as tmp:
        tmp = Path(tmp)
        docs = tmp / "documents"
        write_documents(t, docs)
        views = list(demonstrations(t, docs))
        bank = {}
        meta = {}

        def cache(new):
            nonlocal prepare_time
            began = perf_counter()
            net.eval()
            net.checkpointing = False
            for view in new:
                guard.check()
                uid = view["id"]
                assert uid not in bank
                prompt, target, focus, wrong = encode(tok, view)
                ids = mx.array([prompt + target[:-1]])
                hidden = net.prefix_hidden(ids)[0]
                saved = tmp / (str(len(bank)) + ".safetensors")
                mx.save_safetensors(str(saved), {"hidden": hidden})
                bank[uid] = (saved, len(prompt) - 1, target)
                close = tok.convert_tokens_to_ids("</think>")
                n = (
                    sum(
                        bool(tok.decode([v]).strip())
                        for v in target[: target.index(close)]
                    )
                    if close in target
                    else 0
                )
                meta[uid] = (focus, wrong, n)
            prepare_time += perf_counter() - began

        cache(views)
        for phase in range(2):
            lo = phase * (s["updates"] // 2)
            hi = lo + s["updates"] // 2
            if (
                not (root / f"student-round-{phase}.json.gz").exists()
                and state["step"] != lo
            ):
                raise ValueError(
                    "Missing original on-policy trajectories for recovered weights"
                )
            began = perf_counter()
            peak = max(peak, mx.get_peak_memory())
            collect(net, tok, s, t, docs, root, phase, guard)
            peak = max(
                peak,
                *(
                    q["peak"]
                    for r in unpack(root / f"student-round-{phase}.json.gz")
                    for q in r["requests"]
                ),
            )
            roll_time += perf_counter() - began
            extra = experiences(t, root, tok, docs, s, phase, check=True)
            if (
                len({v["id"].split("-round-")[0] for v in extra})
                < s["min_correction_sources"]
            ):
                raise InterruptedError(
                    f"Insufficient distinct real learner errors for phase {phase}"
                )
            cache(extra)
            views.extend(extra)
            phase_stats.append(
                {
                    "phase": phase,
                    "corrections": len(extra),
                    "with_generated_reasoning": sum(
                        v["completed_thinking"] for v in extra
                    ),
                    "with_real_alternative_token": sum(
                        meta[v["id"]][1] is not None for v in extra
                    ),
                    "wrong_actions_with_successful_callback": sum(
                        v["actual_valid_return"] for v in extra
                    ),
                }
            )
            groups = {
                kind: [v["id"] for v in views if v["kind"] == kind]
                for kind in ("correction", "action", "answer", "retained")
            }
            rng = random.Random(s["fit_seed"] + phase)
            net.train()
            net.checkpointing = True
            for step in range(lo + 1, hi + 1):
                selected = [
                    (rng.choice(groups[kind]), kind, weight)
                    for kind, weight in zip(
                        groups, (0.40, 0.20, 0.25, 0.15), strict=True
                    )
                ]
                if step <= state["step"]:
                    continue
                guard.check()
                grads = None
                objective = 0.0
                for key, kind, weight in selected:
                    focus, wrong, n_reason = meta[key]
                    local, g = grad(
                        net,
                        *pack([bank[key]]),
                        mx.array(focus),
                        *(wrong if wrong else (-1, 0)),
                    )
                    grads = (
                        tree_map(
                            lambda x, scale=weight: x.astype(mx.float32) * scale, g
                        )
                        if grads is None
                        else tree_map(
                            lambda x, y, scale=weight: x + y.astype(mx.float32) * scale,
                            grads,
                            g,
                        )
                    )
                    mx.eval(local, grads)
                    objective += weight * local.item()
                    draws[kind] += 1
                    state["tokens"] += len(bank[key][2])
                    state["proof_tokens"] += n_reason
                    state["contrast"] += int(wrong is not None)
                if not math.isfinite(objective):
                    raise InterruptedError("Nonfinite observed-history training loss")
                adam.update(net, grads)
                mx.eval(net.trainable_parameters(), adam.state)
                if step == 1 or step % 100 == 0:
                    rec = {"step": step, "loss": objective}
                    state["loss"].append(rec)
                    print("Observed-history update", rec, flush=True)
                if step % 100 == 0:
                    state.update(
                        step=step,
                        seconds=prior_time + perf_counter() - start,
                        draws=dict(draws),
                    )
                    checkpoint.write(folder, net, adam, state)
        packed(root / "training-views.json.gz", views)
    weights = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(net.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "weights": digest(weights),
            "training": digest(Path(s["training"])),
            "views": digest(root / "training-views.json.gz"),
            "passes": 2,
            "trainable": count(net, True),
            "turns": len(views),
            "targets": state["tokens"],
            "explanation_targets": state["proof_tokens"],
            "contrast_draws": state["contrast"],
            "updates": s["updates"],
            "draws": dict(draws),
            "loss": state["loss"],
            "phases": phase_stats,
            "prepare_seconds": prepare_time,
            "rollin_seconds": roll_time,
            "entire_fit_seconds": state["seconds"],
            "peak": max(peak, mx.get_peak_memory()),
        },
    )
