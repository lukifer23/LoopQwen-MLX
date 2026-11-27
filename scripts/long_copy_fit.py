"""Teach a recurrent pointer to distinguish two real releases while rehearsing short conversations."""

import argparse
import math
import random
from pathlib import Path
from statistics import mean
from time import monotonic

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import psutil
import subprocess

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.table_reentry_fit import apply
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.source_copy import SourceMixture
from scripts.changed_crate_sources import OUT as SOURCES
from scripts.long_copy_features import PLAN as EXTRACT_PLAN, OUT as FEATURES, ROOT
from scripts.register_source_copy import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ORIGINAL = Path("research/diagnostics/long-copy-fit.json")
PLAN = Path("research/diagnostics/long-copy-fit-runtime-02.json")
OUT = Path("docs/measurements/long-copy-fit")
SEED = 993407


def spec():
    parent = RESULT / FIT
    for p in (SOURCES, FEATURES, EXTRACT_PLAN, parent / "checksums.json"):
        committed(p)
    data = read(SOURCES)
    assert read(FEATURES)["passed"]
    return dict(
        supersedes=digest(ORIGINAL),
        repair="The installed MLX ArrayAt lacks set. Use a position mask to replace one label. The first invocation stopped in the initial training-source assessment, before any optimizer update or reserved-name access. Charge 30 seconds against the original limit.",
        seed=SEED,
        features=digest(FEATURES),
        extraction=digest(EXTRACT_PLAN),
        source=digest(SOURCES),
        model=digest(parent / "checksums.json"),
        implementation=digest(Path(__file__)),
        epochs=4,
        lr=0.00015,
        train=data["training"],
        held=data["reserved"],
        limits=dict(seconds=2170, peak_gib=3, available_gib=1.5),
        protocol="Train only the shared source address and scalar gate in the previously fitted two-read pointer generator. Preserve all original backbone, recurrent, language-correction and remaining head tensors exactly. For each of 80 previously unused package names use both real release versions, with no truncated prefix. Optimize the mean whole visible version and turn terminator likelihood, 0.5 times the two binary likelihood losses contrasting the first real divergent token with that of the other release and a nearby crate, and 0.25 times the sum of true-span NLL and relative true-vs-nearby span loss averaged over visible version decisions. Rehearse a seeded 16-token slice of a previously supervised short conversation in the same update, with NLL plus .25 squared target-log-probability deviation from the frozen parent. Train exactly 640 single-long and single-short paired updates in seeded order. Reserve the parent's original 72 short probes from rehearsal. Stop at 2200 native seconds, 3 GiB allocation or memory pressure. Require complete finite gradients, mean training NLL gain >.03, training address gain >.2, at least eight training packages correct on both source versions against the alternative, and mean short probe loss increase <=.03. Seal weights before scoring reserved packages. Hold fixed all 24 fresh names and both versions. Require positive held NLL gain >=.10, mean held source-location gain >=.10, at least four new net dual-version successes with <=2 lost pairs against initialization, and a positive held second-visit NLL advantage. Report elapsed time, actual memory and losses. Only a passing result can motivate a prospective free-generation comparison on fresh independent documents; this is teacher-forced two-release discrimination, not release accuracy. Previous closed held names and causal-release panels never enter the run.",
    )


def check_guard(start):
    return (
        monotonic() - start < 2170
        and mx.get_peak_memory() < 3 * 2**30
        and psutil.virtual_memory().available > 1.5 * 2**30
        and subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
        ).strip()
        == "1"
    )


def terms(head, project, h, values, ids, pos, labels, span, foil, alternatives):
    logp, attention, gate = head.distributions(h, values, pos, project)

    def selected(y):
        vocab = mx.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        mass = (attention * (ids[:, None, :] == y[..., None])).sum(axis=-1)
        return head.combine(vocab, mass, gate)

    original = selected(labels)
    ce = -original.mean()
    margins = []
    for j, y in alternatives:
        marked = mx.where(mx.arange(labels.shape[1])[None, :] == j, y, labels)
        margins.append(original[0, j] - selected(marked)[0, j])
    a, b = span
    c, d = foil
    source = mx.log(mx.maximum(attention[0, :-1, a:b].sum(axis=-1), 1e-30))
    other = mx.log(mx.maximum(attention[0, :-1, c:d].sum(axis=-1), 1e-30))
    address = -source.mean()
    relative = nn.softplus(other - source).mean()
    binary = nn.softplus(-mx.stack(margins)).mean()
    return (
        ce + 0.5 * binary + 0.25 * (address + relative),
        ce,
        address,
        margins[0],
        margins[1],
    )


def run(freeze):
    s = spec()
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s and not (OUT / "receipt.json").is_file()
    fit = RESULT / FIT
    cfg = read(fit / "registration.json")
    start = monotonic()
    mx.reset_peak_memory()
    actor = SharedQueryAssistant(cfg["checkpoint"], fit)
    net = actor._tool_model
    head = net.step.pointer
    head.freeze()
    head.address.unfreeze()
    head.gate.unfreeze()
    ref = SourceMixture(
        net.text.args.hidden_size, net.step.heads, net.step.dimension, cfg["rank"]
    )
    ref.load_weights(tree_flatten(head.parameters()), strict=True)
    ref.freeze()
    folder = ROOT / digest(EXTRACT_PLAN)
    m = read(FEATURES)
    runs = {}
    for v, p in enumerate(m["versions"]):
        f = folder / p["prefix"]
        assert digest(f) == p["sha256"]
        full = mx.load(str(f))
        for row in p["rows"]:
            runs[(row["id"], v)] = (row, full)
    prior_folder = CACHE / FIT / digest(fit / "registration.json")
    prior = unpack(fit / "feature-index.json.gz")
    probe = [r for r in prior if r["id"] in cfg["probes"]]
    rehearsals = [r for r in prior if r["id"] not in cfg["probes"]]
    assert len(probe) == 72
    nxt = 0.0

    def guard():
        nonlocal nxt
        now = monotonic()
        if now < nxt:
            return
        nxt = now + 0.55
        if not check_guard(start):
            raise InterruptedError("long copy resource limit")

    def inputs(name, v, depth):
        r, p = runs[name, v]
        file = folder / r["file"]
        z = mx.load(str(file))
        values = mx.concatenate([p["values"], z["tail_values"]], axis=2)
        ids = mx.concatenate([p["ids"], z["tail_ids"]], axis=1)
        assert ids.shape[1] == values.shape[2] == r["offset"] + r["positions"]
        return (
            z[depth],
            values,
            ids,
            r["offset"],
            z["labels"],
            r["source"],
            r["foil"],
            r["alternatives"],
        )

    def assess(keys, ptr, depth):
        rows = []
        for name in keys:
            for v in (0, 1):
                guard()
                _, ce, addr, margin, contrast = terms(
                    ptr, net.project, *inputs(name, v, depth)
                )
                rows.append(
                    dict(
                        id=name,
                        version=v,
                        nll=ce.item(),
                        address=addr.item(),
                        margin=margin.item(),
                        local_margin=contrast.item(),
                    )
                )
        return rows

    def short(r, j):
        z = mx.load(str(prior_folder / r["file"]))
        yy = z["labels"][:, j : j + 16]
        return (
            z["two"][:, j : j + 16],
            z["values"],
            z["ids"],
            r["offset"] + j,
            yy,
            net.project,
        )

    def short_probes(ptr):
        rows = []
        for r in probe:
            guard()
            vals = []
            for j in range(0, r["positions"], 16):
                p, _, _ = ptr.selected(*short(r, j))
                vals.append(-p.sum().item())
            rows.append(sum(vals) / r["positions"])
        return rows

    def delta(a, b, k):
        return mean(x[k] - y[k] for x, y in zip(a, b, strict=True))

    def wins(rows):
        src = {r["id"] for r in rows}
        return {k for k in src if all(r["margin"] > 0 for r in rows if r["id"] == k)}

    def loss(ptr, long_args, short_args, old):
        main, ce, addr, marg, local = terms(ptr, net.project, *long_args)
        p, _, _ = ptr.selected(*short_args)
        small = -p.mean() + 0.25 * mx.mean((p - old).square())
        return main + small, ce, addr, small, marg

    gradient = nn.value_and_grad(head, loss)
    opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    result = dict(plan=digest(PLAN), training_passed=False, passed=False, history=[])
    try:
        for r, p in runs.values():
            guard()
            assert digest(folder / r["file"]) == r["sha256"]
        before = assess(s["train"], ref, "two")
        protected = short_probes(ref)
        for epoch in range(s["epochs"]):
            order = [(k, v) for k in s["train"] for v in (0, 1)]
            random.Random(SEED + epoch).shuffle(order)
            for j, (name, v) in enumerate(order):
                guard()
                randomizer = random.Random(fingerprint([SEED, epoch, j]))
                r = randomizer.choice(rehearsals)
                st = randomizer.randint(0, max(0, r["positions"] - 16))
                small = short(r, st)
                teacher = ref.selected(*small)[0]
                mx.eval(teacher)
                (total, ce, addr, sp, margin), g = gradient(
                    head, inputs(name, v, "two"), small, teacher
                )
                norm = apply(head, opt, g)
                vals = [q.item() for q in (total, ce, addr, sp, margin)]
                if not all(map(math.isfinite, [*vals, norm])) or norm <= 0:
                    raise InterruptedError("Invalid gradient")
                result["history"].append(
                    dict(
                        epoch=epoch,
                        crate=name,
                        version=v,
                        short=r["id"],
                        start=st,
                        positions=small[4].size,
                        loss=vals[0],
                        nll=vals[1],
                        address=vals[2],
                        short_loss=vals[3],
                        margin=vals[4],
                        norm=norm,
                    )
                )
            print("full-source copy epoch", epoch + 1, flush=True)
        dest = OUT / "research-policy"
        dest.mkdir(parents=True, exist_ok=True)
        params = dict(tree_flatten(net.step.parameters()))
        old = mx.load(str(fit / "step.safetensors"))
        changed = sorted(
            k for k, a in params.items() if not mx.array_equal(a, old[k]).item()
        )
        assert changed == [
            "pointer.address.weight",
            "pointer.gate.bias",
            "pointer.gate.weight",
        ]
        params_path = dest / "step.safetensors"
        mx.save_safetensors(str(params_path), params)
        policy = {
            k: cfg[k]
            for k in (
                "checkpoint",
                "base_text_sha256",
                "config_sha256",
                "donor",
                "compute_precision",
                "passes",
                "shared_query",
                "revisits",
                "rank",
                "gate_rank",
                "fit_seed",
                "content_read",
                "bounded_write",
                "source_copy",
            )
        }
        persist(
            dest / "registration.json",
            dict(
                **policy,
                id="long-copy-fit",
                experiment=digest(PLAN),
                parent=digest(fit / "checksums.json"),
            ),
        )
        persist(
            dest / "fit.json",
            dict(
                weights=digest(params_path),
                registration=digest(dest / "registration.json"),
                backbone=read(fit / "fit.json")["backbone"],
                updates=len(result["history"]),
            ),
        )
        result["weights"] = digest(params_path)
        after = assess(s["train"], head, "two")
        guard()
        new = short_probes(head)
        observed = dict(
            updates=len(result["history"]) == 640,
            nll_gain=delta(before, after, "nll"),
            address_gain=delta(before, after, "address"),
            dual_correct=len(wins(after)),
            short_increase=mean(y - x for x, y in zip(protected, new, strict=True)),
        )
        result.update(
            training=dict(
                before=before,
                after=after,
                short_before=protected,
                short_after=new,
                **observed,
            ),
            training_passed=observed["updates"]
            and observed["nll_gain"] > 0.03
            and observed["address_gain"] > 0.2
            and observed["dual_correct"] >= 8
            and observed["short_increase"] <= 0.03,
        )
        if result["training_passed"]:
            initial = assess(s["held"], ref, "two")
            one = assess(s["held"], head, "one")
            two = assess(s["held"], head, "two")
            old_wins = wins(initial)
            new_wins = wins(two)
            h = dict(
                parent=initial,
                one=one,
                two=two,
                initial_dual=len(old_wins),
                trained_dual=len(new_wins),
                gained=sorted(new_wins - old_wins),
                lost=sorted(old_wins - new_wins),
                nll_gain=delta(initial, two, "nll"),
                address_gain=delta(initial, two, "address"),
                second_gain=delta(one, two, "nll"),
            )
            result.update(
                held=h,
                passed=h["nll_gain"] >= 0.1
                and h["address_gain"] >= 0.1
                and len(h["gained"]) - len(h["lost"]) >= 4
                and len(h["lost"]) <= 2
                and h["second_gain"] > 0,
            )
        if not check_guard(start):
            raise InterruptedError("long copy resource limit")
    except (InterruptedError, RuntimeError, AssertionError) as e:
        import traceback

        result.update(passed=False, stop=repr(e), trace=traceback.format_exc())
    persist(
        OUT / "receipt.json",
        dict(**result, seconds=monotonic() - start, peak_bytes=mx.get_peak_memory()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
