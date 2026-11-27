"""Fit the two shared recurrent visits to repair a wrong observation and deliver a correct one."""

import argparse
from collections import defaultdict
from functools import partial
import math
from pathlib import Path
import random
from statistics import mean
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import psutil

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import read, digest
from loopqwen.evaluation.generation_eval import advance_cache
from loopqwen.research.table_reentry_fit import apply, accumulate
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.receipt_repair_material import MATERIAL, ROOT, SEED
from scripts.register_source_copy import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ORIGINAL = Path("research/diagnostics/receipt-repair-fit.json")
PLAN = Path("research/diagnostics/receipt-repair-fit-streamed.json")
OUT = Path("docs/measurements/receipt-repair-fit-streamed")
PARENT = RESULT / FIT
FINANCE = ("open", "deliver", "repair", "recovered")
CHUNK = 16
EPOCHS = 2
BATCH = 4


def plan():
    return dict(
        predecessor=digest(ORIGINAL),
        interrupted=digest(Path("docs/measurements/receipt-repair-fit/receipt.json")),
        recovery="Bound the live trunk activation graph with native 128-token cache-state blocks, materialize each retained target-probability slice immediately, and cap the allocator cache at 64 MiB. Recheck chunked native scores against a one-shot forward in the causal gradient regression before launching. Preserve all prompts, target tokens, loss, order, fit and selection gates. Charge the original 69 seconds against the shared maximum.",
        material=digest(MATERIAL),
        bank=digest(ROOT / "inventory.json"),
        parent=digest(PARENT / "checksums.json"),
        implementation=digest(Path(__file__)),
        seed=SEED,
        epochs=EPOCHS,
        batch=BATCH,
        chunk=CHUNK,
        learning_rate=0.00012,
        seconds=2731,
        peak_bytes=3 * 2**30,
        minimum_available=1.5 * 2**30,
        method="Run two shared recurrent reads on the actual full causal prefix for each output token. Cache the unchanged native trunk only. Optimize every shared step and source-mixture tensor, with a frozen donor. Minimize whole-turn normalized target NLL plus 0.65 times mean softplus(1 - correct_vs_plausible_wrong_log_probability_margin) at each first divergent action, value or program token. On nonfinancial rehearsal examples add 0.20 times squared target log-probability change from the unmodified parent. Weight complete rows equally. No tuning or early stopping on the pilot.",
        decision="Reserve all 16 previously identified pilot reports from this intervention. Parent training may have seen the reports. Require all finite updates, lower pilot NLL separately on open, deliver, repair and recovered histories, at least .10 nat improvement in mean contrastive loss for repair and deliver separately, positive two-versus-one-read pilot answer-NLL gain on average, and <=.05 nat average rehearsal pilot NLL increase. A failed gate closes downstream free-generation and product tests for these weights. A successful gate opens free generation on both correct and crossed pilot histories before any fresh direct Base comparison. Archive full or partial results.",
    )


def streamed_arrays(net, row, block=128):
    p, y = row["p"], row["y"]
    cache = net.make_cache()
    for i in range(0, len(p) - 1, block):
        advance_cache(net, mx.array([p[i : min(i + block, len(p) - 1)]]), cache)
    pre, ordinary, keys, values, offset, _ = net.encode(
        mx.array([p[-1:] + y[:-1]]), cache
    )
    assert (
        offset == len(p) - 1
        and keys.shape == values.shape
        and keys.shape[2] == len(p) + len(y) - 1
    )
    data = dict(pre=pre, ordinary=ordinary, keys=keys, values=values)
    mx.eval(data)
    return data


def targets(row):
    labels = row["y"]
    alternatives = [[v, v] for v in labels]
    mask = [[False, False] for _ in labels]
    for j, (pos, token) in enumerate(row["negatives"]):
        alternatives[pos][j] = token
        mask[pos][j] = True
    return dict(
        labels=mx.array([labels]),
        alternatives=mx.array([alternatives]),
        active=mx.array([mask]),
        ids=mx.array([row["p"] + labels[:-1]]),
    )


def probabilities(net, step, z, start, end, offset, visits, differentiating):
    pos = offset + start
    # Native causal masks align queries to the right edge of the supplied K/V.
    limit = offset + min(end, z["pre"].shape[1])
    k, v = z["keys"][:, :, :limit], z["values"][:, :, :limit]
    hidden = net.refine(
        z["pre"][:, start:end],
        z["ordinary"][:, start:end],
        k,
        v,
        pos,
        "causal",
        step=step,
        visits=visits,
        differentiate=differentiating,
    )
    h = net.text.model.norm(hidden)
    pointer = step.pointer
    lang, attention, gate = pointer.distributions(h, v, pos, net.project)

    def selected(labels):
        v = mx.take_along_axis(lang, labels[..., None], axis=-1)[..., 0]
        mass = (attention * (z["ids"][:, None, :limit] == labels[..., None])).sum(
            axis=-1
        )
        return pointer.combine(v, mass, gate)

    positive = selected(z["labels"][:, start:end])
    alternative = z["alternatives"][:, start:end]
    margins = mx.stack(
        [positive - selected(alternative[:, :, i]) for i in range(2)], axis=-1
    )
    return positive, margins


def terms(net, step, z, row, start, end, visits, differentiating):
    logp, margins = probabilities(
        net, step, z, start, end, len(row["p"]) - 1, visits, differentiating
    )
    n = len(row["y"])
    penalties = nn.softplus(1 - margins) * z["active"][:, start:end]
    ce = -logp.sum() / n
    contrast = penalties.sum() / max(1, len(row["negatives"]))
    drift = mx.array(0.0)
    if row["group"] not in FINANCE:
        drift = mx.square(logp - z["reference"][:, start:end]).sum() / n
    return ce + 0.65 * contrast + 0.20 * drift, mx.stack((ce, contrast, drift))


def measure():
    committed(PLAN)
    cfg = read(PLAN)
    assert plan() == cfg
    if (OUT / "receipt.json").exists():
        raise FileExistsError("This registered run already has a terminal receipt")
    net = SharedQueryAssistant(
        read(PARENT / "registration.json")["checkpoint"], PARENT
    )._tool_model
    net.step.unfreeze()
    mx.set_cache_limit(64 * 2**20)
    rows = unpack(MATERIAL)
    training = [i for i, r in enumerate(rows) if r["split"] == "train"]
    pilot = [i for i, r in enumerate(rows) if r["split"] == "pilot"]
    log = []
    result = dict(plan=digest(PLAN), passed=False)
    start = monotonic()
    last_check = 0
    mx.reset_peak_memory()

    def guard():
        nonlocal last_check
        now = monotonic()
        if now - start >= cfg["seconds"] or mx.get_peak_memory() >= cfg["peak_bytes"]:
            raise InterruptedError("native time or memory limit")
        if now - last_check > 1:
            last_check = now
            if (
                psutil.virtual_memory().available < cfg["minimum_available"]
                or subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
                ).strip()
                != "1"
            ):
                raise InterruptedError(
                    f"System memory pressure at {psutil.virtual_memory().available / 2**30:.2f} GiB available, {mx.get_active_memory()/2**30:.2f} GiB active"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory() as tmp:

            def path(i):
                return str(Path(tmp) / f"{i}.safetensors")

            for i, r in enumerate(rows):
                guard()
                z = streamed_arrays(net, r)
                z.update(targets(r))
                if r["group"] not in FINANCE:
                    values = []
                    for j in range(0, len(r["y"]), CHUNK):
                        result_slice = probabilities(
                            net, net.step, z, j, j + CHUNK, len(r["p"]) - 1, 2, False
                        )[0]
                        mx.eval(result_slice)
                        values.append(result_slice)
                    z["reference"] = mx.concatenate(values, axis=1)
                mx.save_safetensors(path(i), z)
                if (i + 1) % 200 == 0:
                    print("causal histories", i + 1, flush=True)

            def probe(visits):
                scores = {}
                for i in pilot:
                    r = rows[i]
                    z = mx.load(path(i))
                    metrics = mx.zeros((3,))
                    for j in range(0, len(r["y"]), CHUNK):
                        guard()
                        _, v = terms(net, net.step, z, r, j, j + CHUNK, visits, False)
                        metrics = metrics + v
                        mx.eval(metrics)
                    scores[r["id"]] = metrics.tolist()
                return scores

            before = probe(2)
            result["before"] = before
            grad = nn.value_and_grad(net.step, partial(terms, net))
            opt = optimizers.AdamW(
                learning_rate=cfg["learning_rate"], weight_decay=0.01
            )
            for epoch in range(EPOCHS):
                order = list(training)
                random.Random(SEED + epoch).shuffle(order)
                for offset in range(0, len(order), BATCH):
                    batch = order[offset : offset + BATCH]
                    gs = None
                    loss = 0.0
                    for i in batch:
                        r = rows[i]
                        z = mx.load(path(i))
                        for j in range(0, len(r["y"]), CHUNK):
                            guard()
                            (v, _), g = grad(net.step, z, r, j, j + CHUNK, 2, True)
                            gs = accumulate(gs, g, 1 / len(batch))
                            mx.eval(gs, v)
                            loss += v.item() / len(batch)
                    norm = apply(net.step, opt, gs)
                    if not math.isfinite(loss) or not 0 < norm < math.inf:
                        raise InterruptedError("invalid optimizer update")
                    log.append(
                        dict(
                            epoch=epoch,
                            rows=[rows[i]["id"] for i in batch],
                            loss=loss,
                            gradient=norm,
                        )
                    )
                    if len(log) % 50 == 0:
                        print("repair updates", len(log), flush=True)
            policy = OUT / "research-policy"
            policy.mkdir(exist_ok=True)
            params = dict(tree_flatten(net.step.parameters()))
            mx.save_safetensors(str(policy / "step.safetensors"), params)
            source = read(PARENT / "registration.json")
            identity = {
                k: source[k]
                for k in (
                    "checkpoint",
                    "base_text_sha256",
                    "config_sha256",
                    "donor",
                    "passes",
                    "shared_query",
                    "revisits",
                    "rank",
                    "gate_rank",
                    "compute_precision",
                    "content_read",
                    "bounded_write",
                    "source_copy",
                    "fit_seed",
                )
            }
            persist(
                policy / "registration.json",
                dict(
                    **identity,
                    id="receipt-repair-fit",
                    experiment=digest(PLAN),
                    parent=cfg["parent"],
                ),
            )
            persist(
                policy / "fit.json",
                dict(
                    weights=digest(policy / "step.safetensors"),
                    registration=digest(policy / "registration.json"),
                    backbone=read(PARENT / "fit.json")["backbone"],
                    updates=len(log),
                ),
            )
            after = probe(2)
            one = probe(1)
            by_group = defaultdict(list)
            for i in pilot:
                r = rows[i]
                by_group[r["group"]].append(r["id"])

            def difference(a, b, keys, col):
                return mean(a[k][col] - b[k][col] for k in keys)

            gains = {
                g: difference(before, after, keys, 0) for g, keys in by_group.items()
            }
            preference = {g: difference(before, after, by_group[g], 1) for g in FINANCE}
            depth = mean(difference(one, after, keys, 0) for keys in by_group.values())
            gates = dict(
                updates=len(log) == EPOCHS * math.ceil(len(training) / BATCH),
                financial=all(gains[g] > 0 for g in FINANCE),
                repair=preference["repair"] > 0.10,
                delivery=preference["deliver"] > 0.10,
                depth=depth > 0,
                retention=mean(v for g, v in gains.items() if g not in FINANCE)
                >= -0.05,
            )
            result.update(
                passed=all(gates.values()),
                gates=gates,
                gains=gains,
                preference=preference,
                depth=depth,
                after=after,
                one=one,
                weights=digest(policy / "step.safetensors"),
            )
            guard()
    except (
        InterruptedError,
        MemoryError,
        RuntimeError,
        AssertionError,
        ValueError,
    ) as e:
        import traceback

        result.update(passed=False, stop=repr(e), trace=traceback.format_exc())
    packed(OUT / "updates.json.gz", log)
    persist(
        OUT / "receipt.json",
        dict(
            **result,
            updates=len(log),
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
        ),
    )
    print(
        "terminal",
        result["passed"],
        result.get("gates", result.get("stop")),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    if p.parse_args().action == "freeze":
        committed(MATERIAL)
        persist(PLAN, plan())
    else:
        measure()
