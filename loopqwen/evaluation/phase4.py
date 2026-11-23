"""Score registered development checkpoints and the untouched Base on identical prompts."""

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from .. import checkpoint
from ..data import retention3, v2, v3
from ..identity import fingerprint, dependencies
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


def evaluate(seed, budgets=(25000, 100000, 250000)):
    reg = read(STUDY)
    prose = [json.loads(line) for line in (retention3.ROOT / "validation.jsonl").open()]
    streams = {
        (s, p): v2.rows(s, p, v3.ROOT)
        for s in ("development", "development_ood")
        for p in ("answer", "trace")
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
        v3=digest(v3.ROOT / "manifest.json"),
        prose=fingerprint(prose),
        seed=seed,
        backend="mlx-metal",
    )
    handles = {}

    def score(arm, budget, k, gen):
        name = f"{arm}-{budget}-k{k}"
        small = budget == 25000
        streams_here = {key: val[:30] if small else val for key, val in streams.items()}
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
            run["likelihood"] = validation_nll(model, items)
            if split == "development" and protocol == "trace":
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
                refinement=refine_stats(
                    model, tok, streams_here["development", "trace"]
                ),
            ),
        )
        print(name, [(p, r["correct"]) for p, r in summaries.items()], flush=True)
        del model
        gc.collect()
        mx.clear_cache()

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
            left, right = read(handles[lhs]), read(handles[rhs])
            pairs[lhs + "/" + rhs] = {
                s: compare(r["records"], right["scores"][s]["records"])
                for s, r in left["scores"].items()
            }
    save(OUT / f"evaluation-{seed}.json", dict(records=handles, pairs=pairs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    evaluate(p.parse_args().seed)
