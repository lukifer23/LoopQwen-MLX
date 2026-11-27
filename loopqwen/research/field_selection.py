"""Train real lookup roles and discriminate three human judgments from the same table."""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from .. import checkpoint
from ..evaluation.mechanism import packed
from ..io import digest, read, save
from ..model_loader import count
from . import reasoned_evidence as comparison
from .file_evidence_fit import pack, projected
from .reasoned_evidence_fit import LABELS

NAME = "field-selection-01"


def turns(s, root):
    from scripts.field_selection_material import training_views

    with TemporaryDirectory() as tmp:
        return training_views(read(s["training"]), Path(tmp))


def qualification(s):
    p = read(s["material"])
    t = read(s["training"])
    return {
        "material": digest(Path(s["material"])),
        "training": digest(Path(s["training"])),
        "training_tables": len(t["tables"]),
        "held_tables": len(p["tables"]),
        "role_tables": len(t["roles"]),
        "published_packages": len(p["wheels"]),
        "training_packages": len({w["package"] for w in t["packages"]}),
        "held_human_field_votes": min(
            q["field_votes"] for w in p["tables"] for q in w["claims"]
        ),
        "held_human_verdict_votes": min(
            q["verdict_votes"] for w in p["tables"] for q in w["claims"]
        ),
    }


def qualify(s, tok, root, guard):
    from scripts.field_selection_material import build

    build()
    data = qualification(s)
    save(root / "source-qualification.json", data)
    if not (
        data["training_tables"] == 128
        and data["held_tables"] == 12
        and data["role_tables"] == 96
        and data["held_human_field_votes"] >= 3
        and data["held_human_verdict_votes"] >= 4
        and data["published_packages"] == 8
    ):
        raise InterruptedError(
            "Insufficient grounded evidence or disjoint tool material"
        )
    guard.check()


def draw_groups(examples, bank, seed):
    by_table = defaultdict(lambda: defaultdict(dict))
    by_kind = defaultdict(list)
    rng = random.Random(seed)
    for w, sample in zip(examples, bank, strict=True):
        kind = w["kind"]
        by_kind[kind].append(sample)
        if kind in ("inline", "file"):
            by_table[w["group"]][kind][w["label"]] = sample
    tables = list(by_table.values())
    while True:
        group = rng.choice(tables)[rng.choice(("inline", "file"))]
        action = rng.choice(("opening", "direction", "repair"))
        package = rng.choice(("package_open", "package_answer"))
        yield [
            (tuple(group[k] for k in LABELS), 0.60, "triplet"),
            ((rng.choice(by_kind[action]),), 0.20, action),
            ((rng.choice(by_kind[package]),), 0.12, package),
            ((rng.choice(by_kind["rehearsal"]),), 0.08, "rehearsal"),
        ]


def positions(batch, tok):
    close = tok.convert_tokens_to_ids("</think>")
    reason, first = [], []
    offset = 0
    for (_, _, ids), label in zip(batch, LABELS, strict=True):
        cut = ids.index(close)
        reason.extend(offset + j for j in range(cut) if tok.decode([ids[j]]).strip())
        j = next(
            j for j in range(cut + 1, len(ids) - 1) if tok.decode([ids[j]]).strip()
        )
        assert ids[j] == tok.encode(label, add_special_tokens=False)[0]
        first.append(offset + j)
        offset += len(ids)
    return reason, first


def triplet_objective(classes):
    order = mx.arange(3)

    def loss(net, h, r, c, y, reason, first):
        logits = projected(net, h, r, c)
        ce = nn.losses.cross_entropy(logits, y)
        mat = logits[first][:, classes]
        discriminate = (
            nn.losses.cross_entropy(mat, order).mean()
            + nn.losses.cross_entropy(mat.T, order).mean()
        ) / 2
        return (
            0.25 * ce[reason].mean()
            + 0.25 * ce[first].mean()
            + 0.20 * ce.mean()
            + 0.30 * discriminate
        )

    return loss


def fit(net, tok, s, root, guard, *, prepare):
    start = perf_counter()
    examples = turns(s, root)
    path = root / "training-views.json.gz"
    packed(path, examples)
    bank = prepare(net, tok, {"examples": examples}, guard)
    batches = draw_groups(examples, bank, s["fit_seed"])
    classes = mx.array([tok.encode(k, add_special_tokens=False)[0] for k in LABELS])
    assert len(set(classes.tolist())) == 3 and net.sites[net.span.start :] == (
        (23, 0),
        (23, 1),
    )
    contrast = nn.value_and_grad(net, triplet_objective(classes))

    def ordinary(net, h, r, c, y):
        return nn.losses.cross_entropy(projected(net, h, r, c), y).mean()

    raw = nn.value_and_grad(net, ordinary)
    net.train()
    net.checkpointing = True
    adam = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    folder = Path("checkpoints/research") / NAME
    folder.mkdir(parents=True, exist_ok=True)
    binding = {
        "registration": digest(root / "registration.json"),
        "views": digest(path),
    }
    state = {
        "step": 0,
        "tokens": 0,
        "proof_tokens": 0,
        "loss": [],
        "seconds": 0,
        "draws": {},
    }
    if (folder / "latest.json").exists():
        state = checkpoint.restore(folder, net, adam)
        assert (
            all(state[k] == v for k, v in binding.items())
            and adam.state["step"].item() == state["step"]
        )
        for _ in range(state["step"]):
            next(batches)
    prefix = perf_counter() - start
    start = perf_counter()
    initial_time = state["seconds"]
    counts = Counter(state["draws"])
    for step in range(state["step"] + 1, s["updates"] + 1):
        guard.check()
        grads = None
        objective = 0
        for batch, weight, kind in next(batches):
            args = pack(batch)
            if kind == "triplet":
                rp, first = positions(batch, tok)
                loss, g = contrast(net, *args, mx.array(rp), mx.array(first))
                state["proof_tokens"] += len(rp)
            else:
                loss, g = raw(net, *args)
            if grads is None:
                grads = tree_map(
                    lambda x, scale=weight: x.astype(mx.float32) * scale, g
                )
            else:
                grads = tree_map(
                    lambda a, b, scale=weight: a + b.astype(mx.float32) * scale,
                    grads,
                    g,
                )
            mx.eval(grads, loss)
            objective += weight * loss.item()
            counts[kind] += len(batch)
            state["tokens"] += sum(len(x[2]) for x in batch)
        if not math.isfinite(objective):
            raise InterruptedError("Nonfinite contrastive training loss")
        adam.update(net, grads)
        mx.eval(net.trainable_parameters(), adam.state)
        guard.check()
        if step == 1 or step % 100 == 0:
            log = {"step": step, "loss": objective}
            state["loss"].append(log)
            print("Field selection", log, flush=True)
            if step % 100 == 0:
                state.update(
                    binding,
                    step=step,
                    draws=dict(counts),
                    seconds=initial_time + perf_counter() - start,
                )
                checkpoint.write(folder, net, adam, state)
    dest = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(dest), dict(tree_flatten(net.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "weights": digest(dest),
            "training": digest(Path(s["training"])),
            "views": digest(path),
            "trainable": count(net, True),
            "passes": 2,
            "updates": s["updates"],
            "targets": state["tokens"],
            "explanation_targets": state["proof_tokens"],
            "turns": len(bank),
            "draws": dict(counts),
            "prepare_seconds": prefix,
            "optimizer_seconds": initial_time + perf_counter() - start,
            "peak": mx.get_peak_memory(),
            "loss": state["loss"],
        },
    )


def decision(p, arms):
    result = comparison.outcome(p, arms)
    a = result["arms"]["candidate"]
    result["gates"].update(
        absolute_accuracy=sum(a["by_label"].values()) >= 30,
        complete_tables=a["whole_stories"] >= 2,
    )
    result["passed"] = all(result["gates"].values())
    return result


def verify(root, d):
    comparison.verify(root, d, name=NAME, make_views=turns, decision_rule=decision)
    assert read(root / "source-qualification.json") == qualification(
        read(root / "registration.json")
    )


if __name__ == "__main__":
    comparison.run(
        name=NAME,
        qualify=qualify,
        trainer=fit,
        decision_rule=decision,
        extra_sources=(
            "loopqwen/research/field_selection.py",
            "scripts/field_selection_material.py",
            "scripts/field_selection_sources.py",
        ),
    )
