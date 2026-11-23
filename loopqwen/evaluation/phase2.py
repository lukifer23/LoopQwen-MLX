"""Manifest-driven V2 study. Historical scores continue to use math_eval."""

import gc
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import psutil

from .. import REVISION
from ..checkpoint import load_trained, location
from ..data.build_data import digest
from ..data.v2 import ROOT, rows
from ..io import read, save
from ..training.losses import validation_nll as nll
from .answers import numeric


def predictions(
    model,
    tok,
    samples,
    limit,
    temperature=0,
    seed=0,
    padded=False,
    batch_size=8,
    detailed=False,
):
    """Batch identical prefix lengths. Charge all executed lanes including finished lanes."""
    grouped = defaultdict(list)
    for r in sorted(samples, key=lambda r: r["prefix"]) if padded else samples:
        grouped[0 if padded else r["prefix"]].append(r)
    output = {}
    durations = []
    mx.random.seed(seed)
    for length, group in grouped.items():
        for i in range(0, len(group), batch_size):
            block = group[i : i + batch_size]
            width = max(r["prefix"] for r in block)
            pads = [width - r["prefix"] for r in block]
            cache = (
                model.make_cache(left_padding=pads)
                if padded and any(pads)
                else model.make_cache()
            )
            x = mx.array(
                [[0] * pad + r["ids"][: r["prefix"]] for pad, r in zip(pads, block)]
            )
            answers = [[] for _ in block]
            done = set()
            mx.synchronize()
            start = perf_counter()
            seeds = [
                int.from_bytes(
                    hashlib.sha256(f"{seed}/{r['id']}".encode()).digest()[:4], "big"
                )
                for r in block
            ]
            for step in range(limit):
                logits = model.project(model.hidden(x, cache)[:, -1, :])
                if padded and step == 0 and any(pads):
                    cache.finish_left_prefill()
                ids = (
                    mx.random.categorical(logits.astype(mx.float32) / temperature)
                    if temperature
                    else mx.argmax(logits, -1)
                )
                if temperature and detailed:
                    ids = mx.stack(
                        [
                            mx.random.categorical(
                                logits[j].astype(mx.float32) / temperature,
                                key=mx.random.key((sample_seed + step) % 2**32),
                            )
                            for j, sample_seed in enumerate(seeds)
                        ]
                    )
                tokens = ids.tolist()
                if step == 0:
                    prefill_seconds = perf_counter() - start
                for j, token in enumerate(tokens):
                    if j not in done:
                        if token in tok.eos_token_ids:
                            done.add(j)
                        else:
                            answers[j].append(token)
                if len(done) == len(block):
                    break
                x = ids[:, None]
            elapsed = perf_counter() - start
            durations.append(
                dict(
                    ids=[r["id"] for r in block],
                    seconds=elapsed,
                    decode_steps=step + 1,
                    prefix=width,
                    prefill_seconds=prefill_seconds,
                    decode_seconds=elapsed - prefill_seconds,
                    executed_positions=len(block) * (width + step),
                )
            )
            for j, (r, ids) in enumerate(zip(block, answers)):
                response = tok.decode(ids)
                pred = numeric(response)
                gold = numeric(r["answer"])
                if gold is None:
                    raise ValueError(r["answer"])
                output[r["id"]] = dict(
                    id=r["id"],
                    prompt=r["prompt"],
                    category=r["category"],
                    complexity=r["complexity"],
                    steps=r["steps"],
                    gold=gold,
                    prediction=pred,
                    correct=pred == gold,
                    text=response,
                    tokens=len(ids),
                    prefix=r["prefix"],
                )
                if detailed:
                    output[r["id"]].update(
                        generated_ids=ids,
                        stop_reason="eos" if j in done else "token_cap",
                        eos_reached=j in done,
                        token_cap_reached=j not in done,
                        request_steps=len(ids) + int(j in done),
                        batch_steps=step + 1,
                        sample_seed=seeds[j] if temperature else None,
                    )
    records = [output[r["id"]] for r in samples]
    return dict(
        correct=sum(r["correct"] for r in records),
        n=len(records),
        seconds=sum(d["seconds"] for d in durations),
        batches=durations,
        records=records,
        max_tokens=limit,
        batching="left-padded" if padded else "equal-length",
        temperature=temperature,
        seed=seed,
    )


def score(model, tok, m, split, language_metrics=None):
    result = {}
    for protocol, limit in m["generation"].items():
        dataset = rows(split, protocol)
        result[protocol] = dict(
            loss=nll(model, dataset[: m["loss_count"]]),
            generations=predictions(
                model,
                tok,
                dataset[: m["generation_count"][protocol]],
                limit,
                padded=m.get("padded", False),
            ),
        )
    language_split = "language_validation" if split == "validation" else "language_test"
    if language_metrics is None:
        language_metrics = {}
    if language_split not in language_metrics:
        language_metrics[language_split] = nll(
            model, rows(language_split, "language")[: m["language_count"]]
        )
    result["language"] = language_metrics[language_split]
    return result


def manifest(name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid experiment ID")
    m = read(Path("experiments") / (name + ".json"))
    if m["revision"] != REVISION or m["data_sha256"] != digest(ROOT / "manifest.json"):
        raise ValueError("Experiment requires its registered model and data revisions")
    return m


def evaluate(name):
    m = manifest(name)
    directory = Path("outputs/phase2") / name
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "lifecycle.json", {"status": "started", "experiment": name})
    for arm in m["runs"]:
        root = Path("checkpoints") / arm
        state = read(location(root) / "state.json")
        if state.get("status") != "completed" or {
            k: v for k, v in state["manifest"].items() if k != "gsm_train"
        } != read(ROOT / "manifest.json"):
            raise ValueError("Incomplete or mismatched run " + arm)
        from ..identity import evaluation_identity

        identity = evaluation_identity(
            m,
            location(root),
            {
                split: digest(ROOT / (split + ".jsonl"))
                for split in set(m["splits"])
                | {
                    "language_validation" if s == "validation" else "language_test"
                    for s in m["splits"]
                }
            },
            {
                "generation": m["generation"],
                "padded": m.get("padded", False),
                "temperature": 0,
            },
            "historical-numeric-v2",
        )
        from ..identity import fingerprint

        path = directory / "revisions" / fingerprint(identity) / (arm + ".json")
        save(
            directory / (arm + ".revision.json"),
            {"path": str(path.relative_to(directory))},
        )
        if path.exists():
            if read(path)["identity"] != identity:
                raise ValueError("Registered evaluation changed " + arm)
            continue
        model, tok, state = load_trained(root)
        # Warm model execution before latency measurement.
        mx.eval(model.project(model.hidden(mx.array([[1, 2, 3]]))[:, -1]))
        mx.reset_peak_memory()
        language_metrics = {}
        scores = {
            split: score(model, tok, m, split, language_metrics)
            for split in m["splits"]
        }
        h = state["history"]
        summary = dict(
            config=state["config"],
            seed=state["config"]["seed"],
            step=state["step"],
            target_tokens=state["loss_tokens"],
            input_tokens=state["input_tokens"],
            update_seconds=sum(r["seconds"] for r in h),
            peak_bytes=max(r["peak_bytes"] for r in h),
            order=state["order_hash"],
            trainable=state["trainable"],
            depth=len(model.sites),
            alpha=h[-1]["alpha"],
            gradient_norm_max=max(r["gradient_norm"] for r in h),
        )
        save(
            path,
            dict(
                identity=identity,
                training=summary,
                scores=scores,
                eval_memory=dict(
                    cache_limit_mib=256,
                    peak=mx.get_peak_memory(),
                    active=mx.get_active_memory(),
                    cached=mx.get_cache_memory(),
                    available=psutil.virtual_memory().available,
                    swap_used=psutil.swap_memory().used,
                ),
            ),
        )
        del model
        gc.collect()
        mx.clear_cache()
        print("Evaluated", arm, flush=True)
    path = report(name)
    save(
        directory / "lifecycle.json",
        {"status": "evaluated", "experiment": name, "runs": m["runs"]},
    )
    return path


def report(name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid experiment ID")
    directory = Path("outputs/phase2") / name
    archive = Path("docs/measurements/phase2") / name / "comparison.md"
    if not directory.exists() and archive.is_file():
        return str(archive)
    m = manifest(name)
    lines = [
        f"# {name}",
        "",
        m["purpose"],
        "",
        "| Run | Split | Updates | Targets | Training s | Answer NLL | Accuracy | Trace NLL | Trace accuracy | Prose NLL | Greedy s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def current(arm):
        pointer = directory / (arm + ".revision.json")
        return read(
            directory / (read(pointer)["path"] if pointer.exists() else arm + ".json")
        )

    for arm in m["runs"]:
        record = current(arm)
        t = record["training"]
        for split, s in record["scores"].items():
            a = s["answer"]
            g = a["generations"]
            trace = s.get("trace", {})
            tr = trace.get("generations", {})
            lines.append(
                f"| {arm} | {split} | {t['step']} | {t['target_tokens']} | {t['update_seconds']:.1f} | {a['loss']['nll']:.4f} | {g['correct']}/{g['n']} | {trace['loss']['nll']:.4f} | {tr['correct']}/{tr['n']} | {s['language']['nll']:.4f} | {g['seconds']:.2f} |"
            )
    for left, right in m.get("matched", []):
        a, b = [current(arm) for arm in (left, right)]
        allowed = {
            "name",
            "passes",
            "curriculum",
            "mode",
            "depth_weights",
            "alpha_schedule",
            "ramp_fraction",
        }
        configs = [
            {k: v for k, v in t["training"]["config"].items() if k not in allowed}
            for t in (a, b)
        ]
        if configs[0] != configs[1]:
            raise ValueError("Dense and loop training policies differ")
        for k in ("order", "seed", "step", "target_tokens", "input_tokens"):
            if a["training"][k] != b["training"][k]:
                raise ValueError(f"Unmatched {k}: {left}, {right}")
        if abs(a["training"]["trainable"] - b["training"]["trainable"]) > 1:
            raise ValueError("Capacity mismatch")
    path = directory / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
