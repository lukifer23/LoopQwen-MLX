"""Manifest-driven V2 study. Historical scores continue to use math_eval."""

import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import REVISION
from ..checkpoint import load_trained, location
from ..data.build_data import digest
from ..data.tokenize import batch
from ..data.v2 import ROOT, rows
from ..io import read, save
from ..training.losses import loss_sum

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
MARKER = r"(?:final answer\s*:\s*|answer\s*:\s*|the answer is\s+|####\s*)"


def numeric(text):
    text = text.split("<|endoftext|>")[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    marked = [i for i, s in enumerate(lines) if re.match(MARKER, s, re.I)]
    if marked and marked[-1] != len(lines) - 1:
        return None
    candidates = [lines[i] for i in marked] if marked else lines
    vals = []
    for line in candidates:
        line = re.sub("^" + MARKER, "", line, flags=re.I).strip().strip("$")
        line = re.sub(r"^\\boxed\{(.*)\}$", r"\1", line)
        line = re.sub(r"^\\frac\{(-?\d+)\}\{(\d+)\}$", r"\1/\2", line)
        m = re.fullmatch(rf"({NUMBER})(?:\s*/\s*({NUMBER}))?[.!]?", line)
        if not m:
            return None
        try:
            n, d = (
                Fraction(x.replace(",", "")) if x else Fraction(1) for x in m.groups()
            )
            vals.append(str(n / d))
        except (ZeroDivisionError, ValueError):
            return None
    return vals[0] if len(set(vals)) == 1 else None


def nll(model, samples):
    model.eval()
    total = labels = 0
    ordered = sorted(samples, key=lambda r: r["input_length"])
    for i in range(0, len(ordered), 4):
        x, p, y = batch(ordered[i : i + 4])
        total += loss_sum(model, x, p, y).item()
        labels += y.size
    return dict(nll=total / labels, perplexity=math.exp(total / labels), tokens=labels)


def predictions(model, tok, samples, limit, temperature=0, seed=0):
    """Batch identical prefix lengths. Charge all executed lanes including finished lanes."""
    grouped = defaultdict(list)
    for r in samples:
        grouped[r["prefix"]].append(r)
    output = {}
    durations = []
    mx.random.seed(seed)
    for length, group in grouped.items():
        for i in range(0, len(group), 8):
            block = group[i : i + 8]
            cache = model.make_cache()
            x = mx.array([r["ids"][:length] for r in block])
            answers = [[] for _ in block]
            done = set()
            mx.synchronize()
            start = perf_counter()
            for step in range(limit):
                logits = model.project(model.hidden(x, cache)[:, -1, :])
                ids = (
                    mx.random.categorical(logits.astype(mx.float32) / temperature)
                    if temperature
                    else mx.argmax(logits, -1)
                )
                for j, token in enumerate(ids.tolist()):
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
                    prefix=length,
                )
            )
            for r, ids in zip(block, answers):
                response = tok.decode(ids)
                pred = numeric(response)
                gold = numeric(r["answer"])
                if gold is None:
                    raise ValueError(r["answer"])
                output[r["id"]] = dict(
                    id=r["id"],
                    category=r["category"],
                    complexity=r["complexity"],
                    steps=r["steps"],
                    gold=gold,
                    prediction=pred,
                    correct=pred == gold,
                    text=response,
                    tokens=len(ids),
                    prefix=length,
                )
    records = [output[r["id"]] for r in samples]
    return dict(
        correct=sum(r["correct"] for r in records),
        n=len(records),
        seconds=sum(d["seconds"] for d in durations),
        batches=durations,
        records=records,
        max_tokens=limit,
        temperature=temperature,
        seed=seed,
    )


def score(model, tok, m, split):
    result = {}
    for protocol, limit in m["generation"].items():
        dataset = rows(split, protocol)
        result[protocol] = dict(
            loss=nll(model, dataset[: m["loss_count"]]),
            generations=predictions(
                model, tok, dataset[: m["generation_count"][protocol]], limit
            ),
        )
    language_split = "language_validation" if split == "validation" else "language_test"
    result["language"] = nll(
        model, rows(language_split, "language")[: m["language_count"]]
    )
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
    for arm in m["runs"]:
        root = Path("checkpoints") / arm
        state = read(location(root) / "state.json")
        if state.get("status") != "completed" or state["manifest"] != read(
            ROOT / "manifest.json"
        ):
            raise ValueError("Incomplete or mismatched run " + arm)
        path = directory / (arm + ".json")
        identity = dict(
            manifest=digest(Path("experiments") / (name + ".json")),
            weights=digest(location(root) / "adapters.safetensors"),
        )
        if path.exists():
            if read(path)["identity"] != identity:
                raise ValueError("Registered evaluation changed " + arm)
            continue
        model, tok, state = load_trained(root)
        # Warm model execution before latency measurement.
        mx.eval(model.project(model.hidden(mx.array([[1, 2, 3]]))[:, -1]))
        scores = {split: score(model, tok, m, split) for split in m["splits"]}
        h = state["history"]
        summary = dict(
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
        save(path, dict(identity=identity, training=summary, scores=scores))
        del model
        mx.clear_cache()
        print("Evaluated", arm, flush=True)
    return report(name)


def report(name):
    m = manifest(name)
    directory = Path("outputs/phase2") / name
    lines = [
        f"# {name}",
        "",
        m["purpose"],
        "",
        "| Run | Split | Updates | Targets | Training s | Answer NLL | Accuracy | Trace NLL | Trace accuracy | Prose NLL | Greedy s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in m["runs"]:
        record = read(directory / (arm + ".json"))
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
        a, b = [read(directory / (arm + ".json")) for arm in (left, right)]
        for k in ("order", "seed", "step", "target_tokens", "input_tokens"):
            if a["training"][k] != b["training"][k]:
                raise ValueError(f"Unmatched {k}: {left}, {right}")
        if abs(a["training"]["trainable"] - b["training"]["trainable"]) > 1:
            raise ValueError("Capacity mismatch")
    path = directory / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
