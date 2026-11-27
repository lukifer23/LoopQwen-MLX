"""Learn to prefer a relevant real tool return over a plausible successful alternative."""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from scripts.valid_return_material import tool_set

from .. import checkpoint
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..model_loader import count
from . import reasoned_evidence as comparison
from .file_evidence_fit import pack, projected
from .native_function_study import executions, summarize
from .valid_return_data import turns

NAME = "valid-return-01"


def qualify(s, tok, root, guard):
    from scripts.valid_return_material import build
    from scripts.valid_return_oracle import run as verify_sources

    build()
    verify_sources()
    t = read(s["training"])
    p = read(s["material"])
    q = {
        "training": digest(Path(s["training"])),
        "material": digest(Path(s["material"])),
        "training_tables": len(t["tables"]),
        "training_events": len(t["agenda"]),
        "rationales": len(t["explanations"]),
        "unseen_packages": len(p["wheels"]),
        "held_human_tables": len(p["tables"]),
        "requests": len(p["rows"]),
    }
    save(root / "source-qualification.json", q)
    if (
        q["training_tables"],
        q["training_events"],
        q["rationales"],
        q["unseen_packages"],
        q["held_human_tables"],
        q["requests"],
    ) != (64, 64, 96, 8, 12, 112):
        raise InterruptedError("Verified preference sources did not meet enrollment.")
    guard.check()


def grouped(views):
    groups = defaultdict(lambda: defaultdict(dict))
    for i, w in enumerate(views):
        groups[w["kind"]][w["source"]].setdefault(w["pair"], {})[w["choice"]] = i
    return {
        k: {s: list(v.values()) for s, v in sources.items()}
        for k, sources in groups.items()
    }


def draws(views, seed):
    pool = grouped(views)
    rng = random.Random(seed)

    def pick(kind):
        src = rng.choice(list(pool[kind]))
        return kind, src, rng.choice(pool[kind][src])

    while True:
        action = "operation" if rng.random() < 0.18 else "field"
        yield [
            (pick(action), 0.44),
            (pick("verdict"), 0.41),
            (pick("repair"), 0.10),
            (pick("rehearsal"), 0.05),
        ]


def fit(net, tok, s, root, guard, *, prepare):
    start = perf_counter()
    net.eval()
    views = turns(s, root)
    viewfile = root / "training-views.json.gz"
    packed(viewfile, views)
    bank = prepare(net, tok, {"examples": views}, guard)
    close = tok.convert_tokens_to_ids("</think>")
    loc = {}
    for i, (_, _, ids) in enumerate(bank):
        if views[i]["kind"] == "verdict":
            cut = ids.index(close)
            explanation = [j for j in range(cut) if tok.decode([ids[j]]).strip()]
            answer = [
                j for j in range(cut + 1, len(ids) - 1) if tok.decode([ids[j]]).strip()
            ]
            assert explanation and answer
            loc[i] = (explanation, answer)
    # Reference differences come from the actual release model, on both complete wire
    # responses to the identical prefix, including the terminal token. All computations use MLX.
    refpath = root / "reference-logodds.json.gz"
    reference = unpack(refpath) if refpath.exists() else {}
    pairs = [
        p
        for k, sources in grouped(views).items()
        if k in ("field", "operation", "verdict")
        for cases in sources.values()
        for p in cases
    ]
    if not reference:
        for pair in pairs:
            guard.check()
            pos, neg = pair["preferred"], pair["rejected"]
            h, r, c, y = pack([bank[pos], bank[neg]])
            ce = nn.losses.cross_entropy(projected(net, h, r, c), y)
            n = len(bank[pos][2])
            ref = (ce[n:].sum() - ce[:n].sum()).item()
            assert math.isfinite(ref)
            reference[views[pos]["pair"]] = ref
        packed(refpath, reference)
    assert set(reference) == {views[p["preferred"]]["pair"] for p in pairs}
    prefix_seconds = perf_counter() - start
    net.train()
    net.checkpointing = True

    def ranked(model, h, r, c, y, n, ref, semantic):
        ce = nn.losses.cross_entropy(projected(model, h, r, c), y)
        delta = ce[n:].sum() - ce[:n].sum()
        likelihood = 0.5 * (ce[:n].mean() + ce[semantic].mean())
        return 0.55 * likelihood + 0.45 * mx.logaddexp(
            mx.array(0.0), -0.25 * (delta - ref)
        )

    def imitation(model, h, r, c, y):
        return nn.losses.cross_entropy(projected(model, h, r, c), y).mean()

    pref_grad = nn.value_and_grad(net, ranked)
    plain_grad = nn.value_and_grad(net, imitation)
    adam = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    folder = Path("checkpoints/research") / NAME
    folder.mkdir(parents=True, exist_ok=True)
    bind = {
        "registration": digest(root / "registration.json"),
        "views": digest(viewfile),
        "reference": digest(refpath),
    }
    state = {
        "step": 0,
        "tokens": 0,
        "proof_tokens": 0,
        "rejected_tokens": 0,
        "loss": [],
        "seconds": 0,
        "draws": {},
    }
    samples = draws(views, s["fit_seed"])
    if (folder / "latest.json").exists():
        state = checkpoint.restore(folder, net, adam)
        assert (
            all(state[k] == v for k, v in bind.items())
            and adam.state["step"].item() == state["step"]
        )
        for _ in range(state["step"]):
            next(samples)
    counts = Counter(state["draws"])
    prior = state["seconds"]
    start = perf_counter()
    for step in range(state["step"] + 1, s["updates"] + 1):
        guard.check()
        objective = 0.0
        grads = None
        for (kind, src, pair), weight in next(samples):
            pos = pair["preferred"]
            ids = bank[pos][2]
            n = len(ids)
            if "rejected" in pair:
                neg = pair["rejected"]
                wrong = bank[neg][2]
                if kind == "verdict":
                    reason, semantic = loc[pos]
                    state["proof_tokens"] += len(reason)
                else:
                    cut = next(i for i, (a, b) in enumerate(zip(ids, wrong)) if a != b)
                    semantic = list(range(cut, n - 1))
                    assert semantic
                loss, g = pref_grad(
                    net,
                    *pack([bank[pos], bank[neg]]),
                    n,
                    mx.array(reference[views[pos]["pair"]]),
                    mx.array(semantic),
                )
                state["rejected_tokens"] += len(wrong)
            else:
                loss, g = plain_grad(net, *pack([bank[pos]]))
            grads = (
                tree_map(lambda x, scale=weight: x.astype(mx.float32) * scale, g)
                if grads is None
                else tree_map(
                    lambda a, b, scale=weight: a + b.astype(mx.float32) * scale,
                    grads,
                    g,
                )
            )
            mx.eval(loss, grads)
            objective += weight * loss.item()
            counts[kind + "." + src] += 1
            state["tokens"] += n
        if not math.isfinite(objective):
            raise InterruptedError("Nonfinite valid-return preference objective")
        adam.update(net, grads)
        mx.eval(net.trainable_parameters(), adam.state)
        guard.check()
        if step == 1 or step % 100 == 0:
            rec = {"step": step, "loss": objective}
            state["loss"].append(rec)
            print("Verified-return fit", rec, flush=True)
            if step % 100 == 0:
                state.update(
                    bind,
                    step=step,
                    draws=dict(counts),
                    seconds=prior + perf_counter() - start,
                )
                checkpoint.write(folder, net, adam, state)
    path = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(net.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "weights": digest(path),
            "training": digest(Path(s["training"])),
            "views": digest(viewfile),
            "reference": digest(refpath),
            "reference_pairs": len(pairs),
            "passes": 2,
            "trainable": count(net, True),
            "turns": len(bank),
            "targets": state["tokens"],
            "explanation_targets": state["proof_tokens"],
            "rejected_tokens": state["rejected_tokens"],
            "updates": s["updates"],
            "draws": dict(counts),
            "loss": state["loss"],
            "prepare_seconds": prefix_seconds,
            "optimizer_seconds": prior + perf_counter() - start,
            "peak": mx.get_peak_memory(),
        },
    )


def score(tok, w, i, trajectory):
    basic = comparison.score(tok, w, i, trajectory)
    if w["family"] != "agenda":
        return basic
    events = iter(executions(trajectory))
    observed = all(
        any(e == required for e in events) for required in w["ordered_events"]
    )
    basic.update(observed=observed, success=bool(basic["correct_answer"] and observed))
    return basic


def decision(panel, arms):
    d = comparison.outcome(panel, arms)
    for side, stream in arms.items():
        group = [r for r in stream if r["family"] == "agenda"]
        stats = summarize(group)
        results = defaultdict(list)
        separate = Counter()
        for w, r in zip(panel["rows"], stream, strict=True):
            if w["family"] == "agenda":
                good = r["requests"][0]["score"]["success"]
                results[w["cluster"]].append(good)
                separate["weekday" if len(w["ordered_events"]) == 2 else "room"] += good
        assert len(results) == 8 and all(len(q) == 2 for q in results.values())
        d["arms"][side].update(
            agenda_pairs=sum(all(v) for v in results.values()),
            agenda_kinds=dict(separate),
        )
        d["arms"][side]["groups"]["agenda"] = stats
    a, b = [d["arms"][k] for k in ("candidate", "control")]
    d["gates"].update(
        absolute_accuracy=sum(a["by_label"].values()) >= 30,
        complete_tables=a["whole_stories"] >= 2,
        practical_calendar=a["agenda_pairs"] >= 4
        and a["agenda_pairs"] > b["agenda_pairs"],
        schedule_completion=a["groups"]["agenda"]["completed"]
        >= b["groups"]["agenda"]["completed"],
    )
    d["passed"] = all(d["gates"].values())
    return d


def verify(root, d):
    comparison.verify(
        root,
        d,
        name=NAME,
        make_views=turns,
        decision_rule=decision,
        score_fn=score,
        tool_catalog=tool_set,
    )
    if not d.get("status"):
        f = read(root / "fit.json")
        ref = unpack(root / "reference-logodds.json.gz")
        assert (
            f["reference"] == digest(root / "reference-logodds.json.gz")
            and len(ref) == f["reference_pairs"]
            and all(math.isfinite(v) for v in ref.values())
        )


if __name__ == "__main__":
    comparison.run(
        name=NAME,
        qualify=qualify,
        trainer=fit,
        decision_rule=decision,
        score_fn=score,
        tool_catalog=tool_set,
        extra_sources=(
            "loopqwen/research/valid_return.py",
            "loopqwen/research/valid_return_data.py",
            "scripts/valid_return_material.py",
            "scripts/agenda_sources.py",
            "scripts/valid_return_oracle.py",
        ),
    )
