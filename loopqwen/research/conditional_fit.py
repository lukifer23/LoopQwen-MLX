"""Fit full response likelihoods and execution-certified program preferences in native MLX."""

import argparse
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
from ..conditional_reentry import resident
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from .conditional_features import CACHE, CODES, feature_root
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .native_function_data import token_pair
from .registered import registered
from .table_policy_learning import visible
from .table_reentry_fit import accumulate, apply, root_for
from .table_worlds import SOURCE, prepare_tables, reward

NAME = "conditional-program-fit-01"


def batches(index, s):
    all_batches = []
    for epoch in range(s["epochs"]):
        examples = [
            (i, k)
            for i, r in enumerate(index)
            for k in range(s["repeat_by_group"].get(r["group"], 1))
        ]
        random.Random(f"{s['fit_seed']}/{epoch}/whole-turn").shuffle(examples)
        batches = []
        current = []
        size = 0
        for i, k in examples:
            r = index[i]
            negative = r["negative"]
            rep = s["repeat_by_group"].get(r["group"], 1)
            choice = (epoch * rep + k) % len(negative) if negative else None
            count = r["positions"] + (
                len(negative[choice]["negative_positions"]) if negative else 0
            )
            if current and size + count > s["micro_positions"]:
                batches.append(current)
                current = []
                size = 0
            current.append((i, choice))
            size += count
        if current:
            batches.append(current)
        group = s["micro_accumulation"]
        all_batches.extend(
            [batches[i : i + group] for i in range(0, len(batches), group)]
        )
    return all_batches


def load_batch(index, where, items):
    positions = []
    labels = []
    weights = []
    pairs = []
    offset = 0
    correct_count = 0
    for i, choice in items:
        rec = index[i]
        a = mx.load(str(where / rec["file"]))
        length = rec["positions"]
        positions.append(a["positive"])
        labels.append(a["labels"])
        weights.extend([1.0] * length)
        if choice is not None:
            good = [offset + j for j in rec["negative"][choice]["positive_positions"]]
        offset += length
        correct_count += length
        if choice is not None:
            n = len(rec["negative"][choice]["negative_positions"])
            positions.append(a[f"negative_{choice}"])
            labels.append(a[f"labels_{choice}"])
            pairs.append((good, list(range(offset, offset + n))))
            weights.extend([0.0] * n)
            offset += n
    # nll(rejected) - nll(chosen) is the mean chosen-minus-rejected log probability.
    coeff = []
    for pos, neg in pairs:
        values = [0.0] * offset
        for j in pos:
            values[j] = -1 / len(pos)
        for j in neg:
            values[j] = 1 / len(neg)
        coeff.append(values)
    return (
        mx.concatenate(positions),
        mx.concatenate(labels),
        mx.array(weights),
        mx.array(coeff, dtype=mx.float32).reshape(len(coeff), offset),
    ), correct_count


def objectives(net, weight, margin):
    def loss(readout, features, labels, supervised, pairs):
        logits = net.project(net.text.model.norm(readout(features))).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, labels, reduction="none")
        total = mx.sum(nll * supervised)
        if pairs.shape[0]:
            odds = pairs @ nll
            total = total + weight * mx.sum(nn.softplus(margin - odds))
        return total

    return loss


def probes(net, index, where, s):
    selected = []
    for group in sorted({r["group"] for r in index}):
        pool = [(i, r) for i, r in enumerate(index) if r["group"] == group]
        selected.extend(
            sorted(
                pool,
                key=lambda pair: fingerprint([s["fit_seed"], "probe", pair[1]["id"]]),
            )[:8]
        )
    output = []
    for i, r in selected:
        a = mx.load(str(where / r["file"]))
        f, y = a["positive"], a["labels"]
        ordinary = f[:, 0 if s["passes"] == 2 else -1]
        native = net.project(net.text.model.norm(net.readout(f))).astype(mx.float32)
        initial = net.project(net.text.model.norm(ordinary)).astype(mx.float32)
        nll = [
            nn.losses.cross_entropy(x, y, reduction="mean").item()
            for x in [initial, native]
        ]
        output.append(
            {
                "id": r["id"],
                "group": r["group"],
                "positions": r["positions"],
                "ordinary": nll[0],
                "trained": nll[1],
            }
        )
    return output


def executions(net, tok, rows, s, guard):
    pool = [r for r in rows if r["group"] == "program"]
    chosen = sorted(
        pool, key=lambda r: fingerprint([s["fit_seed"], "free-action", r["id"]])
    )[:24]
    worlds = {r["id"]: r for r in read(SOURCE)["cohorts"]["train"]}
    cohort = [worlds[r["id"].removesuffix("-program")] for r in chosen]
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    results = []
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, cohort)
        for row, w in zip(chosen, cohort, strict=True):
            p, _ = token_pair(tok, row)
            ids = mx.array([p])
            cache = net.make_cache()
            tokens = []
            for j in range(s["action_cap"]):
                if j % 16 == 0:
                    guard.check()
                logits = net.project(net.last_hidden(ids, cache))
                token = mx.argmax(logits[0, -1]).item()
                tokens.append(token)
                if token in stops:
                    break
                ids = mx.array([[token]])
            ans, closed = visible(tok, {"tokens": tokens}, stops)
            result = reward(ans, closed, row["tools"], w, folder)
            results.append(
                {
                    "id": w["id"],
                    "prefix": p,
                    "tokens": tokens,
                    "visible": ans,
                    "closed": closed,
                    "result": result,
                }
            )
    return results


def decision(s, log, prob, execs):
    group = defaultdict(list)
    for r in prob:
        group[r["group"]].append(r["ordinary"] - r["trained"])
    avg = {k: mean(v) for k, v in group.items()}
    good = [
        r["result"]["statement"]
        for r in execs
        if len(r["result"]["results"]) == 2
        and all(x["correct"] for x in r["result"]["results"])
    ]
    count = sum(r["targets"] for r in log)
    neg = sum(r["preferences"] for r in log)
    gates = {
        "full_exposure": count == s["expected_targets"]
        and neg == s["expected_preferences"],
        "mean_likelihood": mean(avg.values()) >= 0.08,
        "broad_likelihood": min(
            avg[g] for g in ("instruction", "arithmetic", "explanation", "legacy")
        )
        >= -0.02,
        "free_sql": len(good) >= 6 and len(set(good)) >= 4,
        "gradients": all(
            math.isfinite(r["loss"]) and math.isfinite(r["norm"]) and r["norm"] > 0
            for r in log
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "updates": len(log),
        "all_response_positions": count,
        "preference_presentations": neg,
        "group_mean_nll_gain": avg,
        "training_sql_both_correct": len(good),
        "distinct_correct_programs": len(set(good)),
    }


def train(root, s, guard):
    parent = Path("research/results") / s["feature_stage"]
    spec = registered(s["feature_stage"])
    if not read(parent / "decision.json")["passed"]:
        raise InterruptedError("The feature stage missed its parity or exposure gate")
    identity = read(parent / "backbone.json")
    net = resident(s["checkpoint"], root, s, identity)
    idx = unpack(parent / "feature-index.json.gz")
    assert len(idx) == s["examples"]
    where = feature_root(spec)
    for r in idx:
        guard.check()
        if digest(where / r["file"]) != r["sha256"]:
            raise ValueError("A native feature file differs from the sealed source")
    steps = batches(idx, s)
    directory = CACHE / s["id"]
    directory.mkdir(exist_ok=True, parents=True)
    opt = optim.AdamW(s["lr"], weight_decay=0.01)
    obj = objectives(net, s["preference_weight"], s["margin"])
    grad = nn.value_and_grad(net.readout, obj)
    identity_hash = digest(root / "registration.json")
    state = {"step": 0, "log": [], "registration": identity_hash}
    if (directory / "latest.json").exists():
        state = checkpoint.restore(directory, net.readout, opt)
        assert state["registration"] == identity_hash
    for number, micros in enumerate(steps, 1):
        if number <= state["step"]:
            continue
        total = 0
        count = 0
        loss_sum = 0
        pref_count = 0
        g = None
        for micro in micros:
            guard.check()
            inputs, n = load_batch(idx, where, micro)
            v, gradient = grad(net.readout, *inputs)
            g = accumulate(g, gradient, 1)
            mx.eval(v, g)
            loss_sum += v.item()
            total += n
            count += len(micro)
            pref_count += inputs[-1].shape[0]
        norm = apply(net.readout, opt, tree_map(lambda x, n=total: x / n, g))
        if not math.isfinite(loss_sum):
            raise InterruptedError("Nonfinite full-vocabulary fit")
        state["step"] = number
        state["log"].append(
            {
                "step": number,
                "targets": total,
                "examples": count,
                "preferences": pref_count,
                "loss": loss_sum / total,
                "norm": norm,
            }
        )
        if number % 64 == 0 or number == len(steps):
            checkpoint.write(directory, net.readout, opt, state)
            print(
                "Full-response update",
                number,
                "of",
                len(steps),
                "positions",
                sum(r["targets"] for r in state["log"]),
                flush=True,
            )
    saved = checkpoint_location(directory)
    path = root / "readout.safetensors"
    shutil.copyfile(saved / "adapters.safetensors", path)
    shutil.copyfile(saved / "optimizer.safetensors", root / "optimizer.safetensors")
    packed(root / "training-log.json.gz", state["log"])
    save(
        root / "fit.json",
        {
            "registration": identity_hash,
            "weights": digest(path),
            "backbone": identity,
            "feature_archive": digest(parent / "checksums.json"),
            "trainable": sum(v.size for _, v in tree_flatten(net.readout.parameters())),
            "updates": state["step"],
        },
    )
    prob = probes(net, idx, where, s)
    save(root / "training-probes.json", prob)
    execs = executions(net, tokenizer(s), unpack(Path(s["training"])), s, guard)
    packed(root / "free-sql.json.gz", execs)
    return decision(s, state["log"], prob, execs)


def run(name):
    s = registered(name)
    root = root_for(s)
    capture(root, (*CODES, "loopqwen/research/conditional_fit.py"))
    guard = Budget(name, s["limits"])
    try:
        d = train(root, s, guard)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {"passed": False, "status": "interrupted", "reason": str(e)}
        local = CACHE / name
        if (local / "latest.json").is_file():
            cp = checkpoint_location(local)
            for fn in ("adapters.safetensors", "optimizer.safetensors", "state.json"):
                shutil.copyfile(cp / fn, root / ("partial-" + fn))
            d["durable_updates"] = read(cp / "state.json")["step"]
    finally:
        guard.close()
    save(root / "decision.json", d)


def verify(root, d):
    s = read(root / "registration.json")
    assert s == registered(s["id"])
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        if "durable_updates" in d:
            assert d["durable_updates"] == read(root / "partial-state.json")["step"]
        return
    parent = Path("research/results") / s["feature_stage"]
    fit = read(root / "fit.json")
    assert (
        fit["weights"] == digest(root / "readout.safetensors")
        and fit["registration"] == digest(root / "registration.json")
        and fit["feature_archive"] == digest(parent / "checksums.json")
    )
    idx = unpack(parent / "feature-index.json.gz")
    steps = batches(idx, s)
    log = unpack(root / "training-log.json.gz")
    for n, (line, micros) in enumerate(zip(log, steps, strict=True), 1):
        cases = [pair for micro in micros for pair in micro]
        assert (
            line["targets"] == sum(idx[i]["positions"] for i, _ in cases)
            and line["preferences"] == sum(j is not None for _, j in cases)
            and line["examples"] == len(cases)
            and n == line["step"]
        )
    rows = unpack(Path(s["training"]))
    tok = tokenizer(s)
    worlds = {r["id"]: r for r in read(SOURCE)["cohorts"]["train"]}
    actions = unpack(root / "free-sql.json.gz")
    chosen = sorted(
        [r for r in rows if r["group"] == "program"],
        key=lambda r: fingerprint([s["fit_seed"], "free-action", r["id"]]),
    )[:24]
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, [worlds[r["id"]] for r in actions])
        for r, w in zip(actions, chosen, strict=True):
            assert (
                r["id"] + "-program" == w["id"] and token_pair(tok, w)[0] == r["prefix"]
            )
            ans, closed = visible(tok, r, stops)
            assert (
                ans == r["visible"]
                and closed == r["closed"]
                and reward(ans, closed, w["tools"], worlds[r["id"]], folder)
                == r["result"]
            )
    assert decision(s, log, read(root / "training-probes.json"), actions) == d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=NAME)
    run(parser.parse_args().name)
