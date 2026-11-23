import fcntl
import hashlib
import json
import math
import random
import signal
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import psutil
from mlx.utils import tree_map

from .. import checkpoint
from ..config import config, span
from ..data.build_data import ROOT, load_split
from ..data.tokenize import batch, pack, physical_batches
from ..hardware import doctor, gpu_utilization
from ..io import read, save
from ..model_loader import count
from .losses import exit_objective, loss_sum, packed_loss_sum, validation_nll
from .optimizers import optimizer


def evaluate_rows(model, rows, response_only=True):
    model.eval()
    total = tokens = 0
    for row in rows:
        x, pos, y = batch([row], response_only)
        total += loss_sum(model, x, pos, y).item()
        tokens += y.size
    return dict(nll=total / tokens, tokens=tokens, perplexity=math.exp(total / tokens))


def depth(c, step, micro):
    rng = random.Random(f"{c['seed']}/{step}/{micro}")
    if c["curriculum"] == "mixed":
        return rng.choices((1, 2, 3), weights=c["depth_weights"])[0]
    if c["curriculum"] == "gradual" and rng.random() >= min(
        1.0, step / max(1, c["steps"] * c.get("ramp_fraction", 1 / 3))
    ):
        return 1
    return c["passes"]


def train(path, until=None):
    c = config(path)
    root = Path("checkpoints") / c["name"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if c["dataset"] == "pilot":
            return _train(c, root, until)
        progress = root / "lifecycle.json"
        save(progress, {"status": "started", "run": c["name"]})
        try:
            result = _train(c, root, until)
        except (Exception, KeyboardInterrupt) as error:
            save(
                progress,
                {
                    "status": "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    "error": repr(error),
                    "resumable": (root / "latest.json").is_file(),
                },
            )
            raise
        generation = checkpoint.location(root)
        status = read(generation / "state.json")["status"]
        save(
            progress,
            {"status": status, "checkpoint": generation.name, "step": result["step"]},
        )
        return result


def _train(c, root, until):
    policy = None
    if c["dataset"] != "pilot":
        from ..data.sampling import prepare

        rows, valid, manifest, policy = prepare(c)
    else:
        manifest = read(ROOT / "manifest.json")
        rows = load_split(c["training_split"])
        random.Random(c["seed"]).shuffle(rows)
        valid = load_split(c["validation_split"])[: c["eval_count"]]
    model, tok, info = checkpoint.setup(c)
    opt = optimizer(c)
    packed = c.get("packing", False)
    custom_loss = "first_exit_weight" in c or "token_weights" in c
    fn = nn.value_and_grad(
        model,
        exit_objective if custom_loss else packed_loss_sum if packed else loss_sum,
    )
    weight_cache = {}
    exit_weight = c.get("first_exit_weight", 0.0)
    weight_policy = c.get("token_weights")
    if weight_policy:
        from ..data.token_classes import token_classes

        for sample in rows:
            key = (sample["id"], sample["protocol"])
            if key not in weight_cache:
                weight_cache[key] = [
                    weight_policy[t] for t in token_classes(sample, tok)
                ]
    state = dict(
        config=c,
        manifest=manifest,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hardware=doctor(),
        model=info,
        trainable=count(model, True),
        step=0,
        cursor=0,
        history=[],
        input_tokens=0,
        loss_tokens=0,
        order_hash="",
        sequence_length=c["sequence_length"],
        batch_size=c["batch_size"],
        checkpointing=c["gradient_checkpointing"],
    )
    if policy:
        package = Path(__file__).resolve().parents[1]
        sources = hashlib.sha256()
        for file in sorted(package.rglob("*.py")):
            sources.update(str(file.relative_to(package)).encode() + b"\0")
            sources.update(hashlib.sha256(file.read_bytes()).digest())
        state["package_sha256"] = sources.hexdigest()
        state["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain", "--", "loopqwen"])
        )
    if c.get("strict_resume"):
        from ..identity import training_identity

        identity = training_identity(c, manifest)
        state["training_identity"] = identity
    if (root / "latest.json").exists():
        state = checkpoint.restore(root, model, opt)
        if c.get("strict_resume") and state.get("training_identity") != identity:
            raise ValueError("Strict continuation rejected: training identity changed")
        if state["config"] != c or state["manifest"] != manifest:
            raise ValueError("Resume config/data mismatch. Use a distinct run name.")
    stop = c["steps"] if until is None else min(c["steps"], until)
    if stop < state["step"]:
        raise ValueError("Requested stop precedes checkpoint")
    consumed_seconds = sum(r["seconds"] for r in state["history"])

    def budget_met():
        return state["loss_tokens"] >= c.get(
            "label_budget", math.inf
        ) or consumed_seconds >= c.get("seconds_budget", math.inf)

    if policy:
        if state.get("sampling", policy) != policy:
            raise ValueError("Resolved V2 sample schedule changed")
        state["sampling"] = policy
        state.setdefault("visits", {})
    canceled = []

    def request_stop(signum, frame):
        canceled.append(signum)

    handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    mx.set_cache_limit(384 * 1024**2)
    mx.reset_peak_memory()
    begin = perf_counter()
    try:
        while state["step"] < stop and not budget_met():
            s = state["step"]
            pressure = (
                psutil.virtual_memory().available < c["min_available_gib"] * 2**30
                or mx.get_peak_memory() > c["memory_target_gib"] * 2**30
            )
            if pressure:
                mx.clear_cache()
                mx.reset_peak_memory()
                if policy:
                    if state["checkpointing"]:
                        if s:
                            checkpoint.write(root, model, opt, state)
                        raise MemoryError(
                            "V2 memory pressure: checkpoint saved; resume after freeing memory"
                        )
                    state["checkpointing"] = True
                elif state["sequence_length"] > 16:
                    state["sequence_length"] = max(16, state["sequence_length"] // 2)
                elif state["batch_size"] > 1:
                    state["batch_size"] = max(1, state["batch_size"] // 2)
                elif not state["checkpointing"]:
                    state["checkpointing"] = True
                else:
                    raise MemoryError(
                        "At minimum batch and length under configured memory pressure. Resume after freeing memory."
                    )
            model.checkpointing = state["checkpointing"]
            model.train()
            mx.random.seed(c["seed"] + s)
            if c["alpha_schedule"] is not None:
                a, b = c["alpha_schedule"]
                model.alpha_override = a + (b - a) * s / max(1, c["steps"] - 1)
            total = grads = None
            denominator = 0
            exit_sums = raw_sums = None
            forward_alpha = model.effective_alpha() if model.span.has_gate else None
            labels = inputs = 0
            ids = []
            sources = Counter()
            categories = Counter()
            microbatches = []
            examples_by_source = Counter()
            inputs_by_source = Counter()
            lengths = []
            padded = 0
            depths = []
            cursor = state["cursor"]
            start = perf_counter()
            for micro in range(c["accumulation"]):
                selected = []
                for _ in range(len(rows) * state["batch_size"]):
                    row = rows[cursor % len(rows)]
                    cursor += 1
                    if len(row["ids"]) - 1 <= state["sequence_length"]:
                        selected.append(row)
                    if len(selected) == state["batch_size"]:
                        break
                if len(selected) != state["batch_size"]:
                    raise ValueError("Sequence length too short for dataset")
                k = depth(c, s, 0 if "max_microbatch_positions" in c else micro)
                model.span = replace(span(c), passes=k)
                depths.append(k)
                for selected in physical_batches(
                    selected, c.get("max_microbatch_positions")
                ):
                    micro_start = perf_counter()
                    arguments = (pack if packed else batch)(
                        selected, c["response_only"]
                    )
                    if custom_loss:
                        weights = (
                            [
                                w
                                for r in selected
                                for w in weight_cache[(r["id"], r["protocol"])]
                            ]
                            if weight_policy
                            else None
                        )
                        (value, components, raw), g = fn(
                            model,
                            *arguments,
                            mx.array(weights) if weights is not None else 1.0,
                            exit_weight,
                        )
                        exit_sums = (
                            components if exit_sums is None else exit_sums + components
                        )
                        raw_sums = raw if raw_sums is None else raw_sums + raw
                        denominator += (
                            sum(weights) if weights is not None else arguments[2].size
                        )
                    else:
                        value, g = fn(model, *arguments)
                        denominator += arguments[2].size
                    y = arguments[2]
                    if policy:
                        for sample in selected:
                            examples_by_source[sample["protocol"]] += 1
                            inputs_by_source[sample["protocol"]] += sample[
                                "input_length"
                            ]
                            sources[sample["protocol"]] += sample["output_length"]
                            categories[sample["category"]] += sample["output_length"]
                            lengths.append(sample["input_length"])
                        padded += arguments[0].size
                    grads = (
                        g if grads is None else tree_map(lambda a, b: a + b, grads, g)
                    )
                    total = value if total is None else total + value
                    mx.eval(grads, total)
                    if c.get("strict_resume"):
                        microbatches.append(
                            dict(
                                shape=list(arguments[0].shape),
                                k=k,
                                scored=y.size,
                                protocols=dict(
                                    Counter(r["protocol"] for r in selected)
                                ),
                                seconds=perf_counter() - micro_start,
                            )
                        )
                    labels += y.size
                    inputs += sum(len(r["ids"]) - 1 for r in selected)
                    ids.extend(r["id"] for r in selected)
            grad_seconds = perf_counter() - start
            grads = tree_map(lambda g: g / denominator, grads)
            gate_gradient = grads.get("alpha_logit")
            old_gate = model.alpha_logit if gate_gradient is not None else None
            grads, norm = optim.clip_grad_norm(grads, c["clip"])
            mean = total.item() / denominator
            magnitude = norm.item()
            if not math.isfinite(mean) or not math.isfinite(magnitude):
                raise FloatingPointError(f"Nonfinite loss or gradient at step {s + 1}")
            t = perf_counter()
            if "gate_lr" in c and max(depths) == 1:
                adapter_grads = {
                    key: value for key, value in grads.items() if key != "alpha_logit"
                }
                opt.optimizers[0].update(model, adapter_grads)
            else:
                opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            elapsed = perf_counter() - start
            row = dict(
                step=s + 1,
                loss=mean,
                gradient_norm=magnitude,
                inputs=inputs,
                labels=labels,
                depths=depths,
                seconds=elapsed,
                grad_seconds=grad_seconds,
                optimizer_s=perf_counter() - t,
                peak_bytes=mx.get_peak_memory(),
                sequence_limit=state["sequence_length"],
                batch_size=state["batch_size"],
            )
            if c.get("max_microbatch_positions") or custom_loss:
                row.update(
                    objective_denominator=denominator,
                    exit_targets=labels * (2 if exit_weight else 1),
                    decoder_site_positions=sum(
                        b["shape"][0]
                        * b["shape"][1]
                        * (
                            len(model.text.layers)
                            + (b["k"] - 1) * model.span.length
                            + (
                                len(model.text.layers)
                                - model.span.start
                                - model.span.length
                                if exit_weight
                                else 0
                            )
                        )
                        for b in microbatches
                    ),
                    gate_active=forward_alpha is not None and max(depths) > 1,
                    alpha_forward=float(forward_alpha)
                    if forward_alpha is not None and max(depths) > 1
                    else None,
                )
                if custom_loss:
                    row.update(
                        exit_ce=(exit_sums / denominator).tolist(),
                        raw_exit_ce=(raw_sums / labels).tolist(),
                        first_exit_weight=exit_weight,
                    )
            consumed_seconds += elapsed
            if policy:
                repeats = 0
                for key in ids:
                    repeats += int(key in state["visits"])
                    state["visits"][key] = state["visits"].get(key, 0) + 1
                if s % 8 == 0:
                    row["device_utilization_percent"] = gpu_utilization()
                row.update(
                    scored_by_source=dict(sources),
                    scored_by_category=dict(categories),
                    lengths=lengths,
                    padding_fraction=1 - inputs / padded,
                    repeats=repeats,
                    rss=psutil.Process().memory_info().rss,
                    available=psutil.virtual_memory().available,
                    swap=psutil.swap_memory().used,
                    lr=opt.learning_rate.item(),
                    alpha=float(model.effective_alpha())
                    if model.span.has_gate
                    else None,
                )
            if c.get("strict_resume"):
                row.update(
                    microbatches=microbatches,
                    examples_by_source=dict(examples_by_source),
                    inputs_by_source=dict(inputs_by_source),
                    executed_positions=padded,
                    checkpointing=state["checkpointing"],
                    mlx_active=mx.get_active_memory(),
                    mlx_cached=mx.get_cache_memory(),
                )
            if gate_gradient is not None:
                row.update(
                    gate_logit=model.alpha_logit.item(),
                    gate_gradient=gate_gradient.item(),
                    gate_delta=(model.alpha_logit - old_gate).item(),
                    gate_lr=(
                        opt.optimizers[1] if "gate_lr" in c else opt
                    ).learning_rate.item(),
                )
            old_targets = state["loss_tokens"]
            crossed = [
                m
                for m in c.get("milestones", [])
                if old_targets < m <= old_targets + labels
            ]
            state.update(
                step=s + 1,
                cursor=cursor,
                input_tokens=state["input_tokens"] + inputs,
                loss_tokens=state["loss_tokens"] + labels,
                order_hash=hashlib.sha256(
                    (state["order_hash"] + "|".join(ids)).encode()
                ).hexdigest(),
                alpha_override=model.alpha_override,
                next_mlx_seed=c["seed"] + s + 1,
            )
            model.span = span(c)
            if (
                crossed
                or (s + 1) % c["eval_interval"] == 0
                or s + 1 == c["steps"]
                or budget_met()
            ):
                val_start = perf_counter()
                if c.get("validation_batch_size", 1) > 1:
                    row["validation"] = validation_nll(
                        model, valid, batch_size=c["validation_batch_size"]
                    )
                else:
                    row["validation"] = evaluate_rows(model, valid, c["response_only"])
                if policy:
                    row["validation_seconds"] = perf_counter() - val_start
            state["history"].append(row)
            if (
                (s + 1) % c["checkpoint_interval"] == 0
                or crossed
                or s + 1 == stop
                or budget_met()
                or canceled
            ):
                state["status"] = (
                    "completed"
                    if s + 1 == c["steps"] or budget_met()
                    else "interrupted"
                    if canceled
                    else "started"
                )
                io_start = perf_counter()
                checkpoint.write(root, model, opt, state)
                if c.get("max_microbatch_positions"):
                    gen = checkpoint.location(root)
                    with (root / "serialization.jsonl").open("a") as file:
                        print(
                            json.dumps(
                                dict(
                                    step=s + 1,
                                    seconds=perf_counter() - io_start,
                                    bytes=sum(p.stat().st_size for p in gen.iterdir()),
                                )
                            ),
                            file=file,
                        )
                for m in crossed:
                    save(root / f"targets-{m}.json", read(root / "latest.json"))
                save(Path("outputs") / (c["name"] + ".json"), state)
                print(
                    f"{c['name']} update {s + 1} loss {mean:.4f}, {inputs / elapsed:.1f} input tok/s, peak {row['peak_bytes'] / 2**30:.2f} GiB",
                    flush=True,
                )
            if canceled:
                raise KeyboardInterrupt("Saved at update boundary")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return dict(
        run=c["name"],
        pretrained_unique=state["model"]["unique_parameters"],
        trainable_adapters=state["trainable"]
        - int("alpha_logit" in model.trainable_parameters()),
        trainable_gates=int("alpha_logit" in model.trainable_parameters()),
        effective_layer_executions=len(model.sites),
        step=state["step"],
        input_tokens=state["input_tokens"],
        loss_tokens=state["loss_tokens"],
        wall_seconds=perf_counter() - begin,
    )
