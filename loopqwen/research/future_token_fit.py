"""Supervise the shared block's first application on a two-token horizon."""

import gc
import random
import shutil
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from .raw_distillation import pad


class Horizon(nn.Module):
    def __init__(self, decoder, width):
        super().__init__()
        self.decoder = decoder
        size = decoder.args.hidden_size
        self.down = nn.Linear(size, width, bias=False)
        self.up = nn.Linear(width, size, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def future_logits(self, first):
        residual = self.up(nn.silu(self.down(self.decoder.text.model.norm(first))))
        return self.decoder.project(
            self.decoder.text.model.norm(first + residual.astype(first.dtype))
        )


def teacher_receipts(spec, root):
    source = Path(spec["teacher_records"])
    inventory = read(source / "checksums.json")
    for name, expected in spec["teacher_files"].items():
        if inventory.get(name) != expected or digest(source / name) != expected:
            raise ValueError("Changed original teacher supervision: " + name)
        dest = root / name
        if not dest.exists():
            shutil.copyfile(source / name, dest)
        if digest(dest) != expected:
            raise ValueError("Different local teacher record: " + name)


def fit(net, rows, balances, spec, teachers, root, budget):
    from .selective_raw import partial_ce

    at = perf_counter()
    width = max(len(w["ids"]) - 1 for w in rows)
    bank = []
    for i in range(0, len(rows), 8):
        h = net.prefix_hidden(pad(rows[i : i + 8]))
        batch = mx.pad(h, ((0, 0), (0, width - h.shape[1]), (0, 0)))
        mx.eval(batch)
        bank.append(batch)
        budget.check()
    bank = mx.concatenate(bank)
    mx.eval(bank)
    prepared = perf_counter() - at
    objective = Horizon(net, spec["future_width"])
    opt = optimizers.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    rng = random.Random(spec["fit_seed"])

    def loss(obj, prefix, positions, gold, future, large, original):
        first = obj.decoder.ordinary(prefix)
        final, _ = obj.decoder.refine(first)
        select = (mx.arange(prefix.shape[0])[:, None], positions)
        z = obj.decoder.project(final[select]).astype(mx.float32)
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        target = -mx.take_along_axis(lp, gold[..., None], axis=-1)[..., 0]
        blend = mx.sigmoid(
            (large["gold_logp"] - original["gold_logp"]) / spec["temperature"]
        )
        teacher = blend * partial_ce(lp, large["ids"], large["mass"]) + (
            1 - blend
        ) * partial_ce(lp, original["ids"], original["mass"])
        primary = (
            (1 - spec["gold_weight"]) * teacher + spec["gold_weight"] * target
        ).mean()
        aux = nn.losses.cross_entropy(
            obj.future_logits(first[select]).astype(mx.float32), future
        ).mean()
        return primary + spec["future_weight"] * aux

    vg = nn.value_and_grad(objective, loss)
    eligible = [
        [j for j, pos in enumerate(r["positions"]) if pos + 2 < len(r["ids"])]
        for r in rows
    ]
    log = []
    started = perf_counter()
    for step in range(spec["updates"]):
        ix = [
            i
            for group in balances
            for i in rng.sample(group, spec["batch"] // len(balances))
        ]
        rng.shuffle(ix)
        js = [rng.sample(eligible[i], spec["per_document"]) for i in ix]
        positions = [
            [rows[i]["positions"][j] for j in inds]
            for i, inds in zip(ix, js, strict=True)
        ]
        labels = [
            mx.array(
                [
                    [rows[i]["ids"][p + ahead] for p in ps]
                    for i, ps in zip(ix, positions, strict=True)
                ]
            )
            for ahead in (1, 2)
        ]
        selection = (mx.array(ix)[:, None], mx.array(js))
        distributions = [
            {k: arr[selection] for k, arr in teacher.items()} for teacher in teachers
        ]
        value, grads = vg(
            objective, bank[mx.array(ix)], mx.array(positions), *labels, *distributions
        )
        grads, _ = optimizers.clip_grad_norm(grads, 1.0)
        opt.update(objective, grads)
        mx.eval(value, objective.trainable_parameters(), opt.state)
        budget.check()
        if step % 200 == 0 or step + 1 == spec["updates"]:
            log.append([step + 1, value.item()])
            print("two-horizon update", *log[-1], flush=True)
    seconds = perf_counter() - started
    weights = dict(tree_flatten(net.trainable_parameters()))
    auxiliary = {
        k: v
        for k, v in tree_flatten(objective.trainable_parameters())
        if not k.startswith("decoder.")
    }
    assert set(auxiliary) == {"down.weight", "up.weight"} and len(weights) == 10
    path = root / "refinement.safetensors"
    head = root / "future-head.safetensors"
    mx.save_safetensors(str(path), weights)
    mx.save_safetensors(str(head), auxiliary)
    seen = spec["updates"] * spec["batch"] * spec["per_document"]
    save(
        root / "fit.json",
        {
            "weights": digest(path),
            "training": digest(Path(spec["training"])),
            "trainable": sum(v.size for v in weights.values()),
            "parameters": sorted(weights),
            "updates": spec["updates"],
            "seen": seen,
            "future_targets": seen,
            "future_head": digest(head),
            "aux_parameters": sum(v.size for v in auxiliary.values()),
            "prepare_seconds": prepared,
            "optimizer_seconds": seconds,
            "loss": log,
        },
    )
    del bank, objective, vg, opt
    gc.collect()
    mx.clear_cache()


def source_decision(decision, data, items):
    sources = defaultdict(lambda: [0.0, 0, 0])
    for w, a, b in zip(items, data["candidate"], data["control"], strict=True):
        if w["suite"] != "fresh":
            continue
        by = sources[w["source"]]
        by[0] += sum(a["nll"]) - sum(b["nll"])
        by[1] += sum(
            int(x == y) - int(z == y)
            for x, z, y in zip(a["ids"], b["ids"], w["ids"][1:], strict=True)
        )
        by[2] += len(a["ids"])
    wins = sum(s[0] < 0 for s in sources.values())
    losses = sum(s[0] > 0 for s in sources.values())
    p = exact_mcnemar(wins, losses)
    decision.update(
        source_totals=dict(sources),
        source_wins=wins,
        source_losses=losses,
        source_p=p,
        passed=bool(
            decision["passed"] and len(sources) == 24 and wins > losses and p <= 0.05
        ),
    )
    return decision


def free_continuations(net, items, indices, budget):
    from mlx_lm.models.cache import KVCache

    from ..identity import fingerprint

    records = []
    for i in indices:
        row = items[i]
        ids = row["ids"][:32]
        cache = net.make_cache()
        cached = []
        fresh = []
        for step in range(8):
            block = ids if not step else cached[-1:]
            z = net.hidden(mx.array([block]), cache)
            cached.append(mx.argmax(net.project(z[:, -1])).item())
            full = net.hidden(mx.array([ids + fresh]))
            fresh.append(mx.argmax(net.project(full[:, -1])).item())
            budget.check()
        sites = [
            [layer, turn, slot.offset, slot.keys_and_values()[0].shape[2]]
            for (layer, turn), slot in cache.slots.items()
            if isinstance(slot, KVCache)
        ]
        records.append(
            {
                "id": row["id"],
                "prefix_sha256": fingerprint(ids),
                "cached": cached,
                "full": fresh,
                "sites": sites,
            }
        )
    return records


def check_continuations(records, items, indices):
    from ..identity import fingerprint

    expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)} | {(23, 1)}
    consistent = True
    for r, ix in zip(records, indices, strict=True):
        assert r["id"] == items[ix]["id"] and r["prefix_sha256"] == fingerprint(
            items[ix]["ids"][:32]
        )
        assert len(r["cached"]) == len(r["full"]) == 8
        assert {(l, t) for l, t, _, _ in r["sites"]} == expected
        consistent &= (
            all(n == stored == 39 for _, _, n, stored in r["sites"])
            and r["cached"] == r["full"]
        )
    return consistent
