"""Score registered development checkpoints and the untouched Base on identical prompts."""

import argparse
import gc
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from .. import checkpoint
from ..data import retention3, v2, v3
from ..identity import fingerprint, dependencies
from ..data.token_classes import token_classes
from ..data.tokenize import batch
from ..io import read, save, digest, checkpoint_location
from ..looped_model import LoopedModel
from ..model_loader import load_base, local_snapshot
from ..recurrence import Span
from ..training.losses import validation_nll
from .identity import runtime
from .phase2 import predictions
from .statistics import paired
from .trace_verifier import verify

OUT = Path("outputs/phase4")
STUDY = Path("experiments/phase4.json")


def compare(a, b):
    delta, report = paired(a, b)
    sample = np.random.default_rng(4081).choice(delta, (8000, len(delta))).mean(axis=1)
    return dict(
        **report,
        item_ci95=np.quantile(sample, [0.025, 0.975]).tolist(),
        verified_change=sum(
            int(r.get("verified", False)) - int(l.get("verified", False))
            for l, r in zip(a, b, strict=True)
        ),
    )


def partition_likelihood(model, tok, items):
    """Score each target once; maintain ordered per-item likelihoods across length groups."""
    groups = Counter()
    counts = Counter()
    per_item = {}
    ordered = sorted(items, key=lambda r: r["input_length"])
    for i in range(0, len(items), 4):
        group = ordered[i : i + 4]
        x, pos, y = batch(group)
        h = model.hidden(x)
        selected = h.reshape(-1, h.shape[-1])[pos]
        values = []
        for j in range(0, y.size, 64):
            scores = mx.fast.cross_entropy(
                model.project(selected[j : j + 64]), y[j : j + 64]
            )
            values.extend(scores.tolist())
        offset = 0
        for r in group:
            labels = token_classes(r, tok)
            totals = Counter()
            for category, value in zip(
                labels, values[offset : offset + len(labels)], strict=True
            ):
                counts[category] += 1
                groups[category] += value
                totals[category] += value
            per_item[r["id"]] = dict(id=r["id"], final_logp=-totals["final"])
            offset += len(labels)
        if offset != y.size:
            raise ValueError("Scored target coverage mismatch")
    total = sum(counts.values())
    nll = sum(groups.values()) / total
    return dict(
        tokens=total,
        nll=nll,
        perplexity=math.exp(nll),
        tokens_by_class=counts,
        nll_by_class={k: v / counts[k] for k, v in groups.items()},
        records=[per_item[r["id"]] for r in items],
    )


def refine_stats(model, tok, samples):
    values = []

    def observe(prev, raw, out):
        x, y, z = [v.astype(mx.float32) for v in (prev, raw, out)]
        alpha = model.effective_alpha()
        precise = (x + alpha * (y - x)).astype(prev.dtype).astype(mx.float32)
        norm = lambda v: mx.sqrt(mx.mean(v * v))
        delta = z - x
        token_rms = mx.sqrt(mx.mean(delta * delta, axis=-1))
        feat_rms = mx.sqrt(mx.mean(delta * delta, axis=(0, 1)))
        values.append(
            dict(
                alpha=float(alpha),
                previous_rms=norm(x).item(),
                candidate_rms=norm(y).item(),
                relative_raw=(norm(y - x) / norm(x)).item(),
                relative_effective=(norm(delta) / norm(x)).item(),
                cosine=(mx.mean(x * z) / (norm(x) * norm(z))).item(),
                rounded_unchanged=mx.mean((z == x).astype(mx.float32)).item(),
                fp32_cast_difference=norm(precise - z).item(),
                max_token_rms=mx.max(token_rms).item(),
                max_feature_rms=mx.max(feat_rms).item(),
            )
        )

    if model.span.has_gate and model.span.passes == 2:
        for row in samples[:4]:
            list(
                model.block_states(mx.array([row["ids"][: row["prefix"]]]), 2, observe)
            )
    return values


def evaluate(seed, budgets=(25000, 100000, 250000), suite="primary"):
    reg = read(STUDY)
    prose = [json.loads(line) for line in (retention3.ROOT / "validation.jsonl").open()]
    data_root = Path("data/v4") if suite == "v4" else v3.ROOT
    splits = (
        ("validation", "ood") if suite == "v4" else ("development", "development_ood")
    )
    streams = {
        (s, p): v2.rows(s, p, data_root) for s in splits for p in ("answer", "trace")
    }
    source = local_snapshot()
    common = dict(
        schema=1,
        registration=digest(STUDY),
        runtime=runtime(),
        dependencies=dependencies(),
        evaluator=digest(Path(__file__)),
        source={p.name: digest(p) for p in source.glob("*.safetensors")},
        v2=digest(v2.ROOT / "manifest.json"),
        evaluation_manifest=digest(data_root / "manifest.json"),
        suite=suite,
        prose=fingerprint(prose),
        seed=seed,
        backend="mlx-metal",
    )
    handles = {}

    def score(arm, budget, k, gen):
        name = f"{arm}-{budget}-k{k}"
        small = budget == 25000
        streams_here = {
            key: val[:60] if suite == "v4" else val[:30] if small else val
            for key, val in streams.items()
        }
        ident = dict(
            **common,
            arm=arm,
            depth=k,
            train_budget=budget,
            items=fingerprint(list(streams_here.values())),
            weights=digest(gen / "adapters.safetensors") if gen else common["source"],
            state=digest(gen / "state.json") if gen else None,
            decoding=reg["generation"],
        )
        path = OUT / "eval" / str(seed) / name / (fingerprint(ident) + ".json")
        handles[name] = str(path)
        if path.exists():
            record = read(path)
            if record["identity"] != ident:
                raise ValueError("Incompatible cached evaluation")
            return
        if gen:
            model, tok, state = checkpoint.load_trained(gen)
            model.span = replace(model.span, passes=k)
        else:
            base, tok, _ = load_base()
            model = LoopedModel(base, Span(passes=1))
        model.eval()
        summaries = {}
        for (split, protocol), items in streams_here.items():
            cap = reg["generation"][protocol + "_cap"]
            predictions(model, tok, items[:1], 4, padded=True)
            run = predictions(
                model, tok, items, cap, padded=True, batch_size=8, detailed=True
            )
            for pred, row in zip(run["records"], items, strict=True):
                if pred["id"] != row["id"]:
                    raise ValueError("Mismatched request identity")
                if protocol == "trace":
                    cert = verify(row, pred["text"], pred["token_cap_reached"])
                    pred.update(
                        verified=cert["verified"],
                        correct=cert["final"] == "correct",
                        verification=cert,
                    )
            run["correct"] = sum(r["correct"] for r in run["records"])
            run["verified"] = sum(r.get("verified", False) for r in run["records"])
            run["likelihood"] = partition_likelihood(model, tok, items)
            if [r["id"] for r in run["records"]] != [
                r["id"] for r in run["likelihood"]["records"]
            ]:
                raise ValueError("Likelihood and generation cohorts differ")
            if split == splits[0] and protocol == "trace":
                sample = predictions(
                    model, tok, items[:8], cap, padded=True, batch_size=1, detailed=True
                )
                run["interactive"] = dict(
                    seconds=sample["seconds"], batches=sample["batches"]
                )
            summaries[split + "/" + protocol] = run
        save(
            path,
            dict(
                identity=ident,
                scores=summaries,
                retention=validation_nll(model, prose),
                refinement=refine_stats(model, tok, streams_here[splits[0], "trace"]),
            ),
        )
        print(name, [(p, r["correct"]) for p, r in summaries.items()], flush=True)
        del model
        gc.collect()
        mx.clear_cache()

    def contrast(left_name, right_name):
        left, right = [
            read(handles[name])["scores"] for name in (left_name, right_name)
        ]
        pairs[left_name + "/" + right_name] = {
            k: compare(v["records"], right[k]["records"]) for k, v in left.items()
        }

    pairs = {}
    if suite in ("boundary", "v4"):
        arms = (
            ("dense_boundary", "joint_boundary")
            if suite == "boundary"
            else ("dense_v4", "joint_v4", "dense_v4_weighted", "joint_v4_weighted")
        )
        for arm in arms:
            root = Path("checkpoints") / f"p4_{seed}_{arm}"
            for k in (1, 2) if arm.startswith("joint") else (1,):
                score(arm, 25000, k, checkpoint_location(root))
        for arm in arms:
            if arm.startswith("joint"):
                contrast(arm + "-25000-k1", arm + "-25000-k2")
                contrast(arm.replace("joint", "dense") + "-25000-k1", arm + "-25000-k2")
        save(
            OUT / f"evaluation-{seed}-{suite}.json", dict(records=handles, pairs=pairs)
        )
        return
    score("base", 0, 1, None)
    for budget in budgets:
        for arm in (
            ("dense", "gated", "joint")
            if budget <= 100000
            else ("dense", "gated", "time")
        ):
            root = (
                Path("checkpoints") / f"p4_{seed}_{'dense' if arm == 'time' else arm}"
            )
            if arm == "dense" and not (root / f"targets-{budget}.json").exists():
                root = Path("checkpoints") / f"p4_{seed}_dense_data"
            gen = (
                checkpoint_location(root)
                if arm == "time"
                else root / read(root / f"targets-{budget}.json")["directory"]
            )
            for k in (1, 2) if arm in ("gated", "joint") else (1,):
                score(arm, budget, k, gen)
    pairs = {}
    for budget in budgets:
        names = ["dense", "time"] if budget == 250000 else ["dense", "gated"]
        main = "gated" if budget == 250000 else "joint"
        for lhs, rhs in [
            (f"{a}-{budget}-k1", f"{main}-{budget}-k{1 if budget == 250000 else 2}")
            for a in names
        ] + [(f"{main}-{budget}-k1", f"{main}-{budget}-k2")]:
            contrast(lhs, rhs)
        if budget < 250000:
            contrast(f"gated-{budget}-k2", f"joint-{budget}-k2")
            contrast(f"gated-{budget}-k1", f"gated-{budget}-k2")
    save(OUT / f"evaluation-{seed}.json", dict(records=handles, pairs=pairs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    p.add_argument("--suite", choices=["primary", "boundary", "v4"], default="primary")
    args = p.parse_args()
    evaluate(args.seed, suite=args.suite)
