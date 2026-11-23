"""Fresh development curves, diverse retention, and a traced latency frontier."""

import gc
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from .. import checkpoint
from ..data import retention3, v2, v3
from ..identity import fingerprint
from ..io import read, save
from .inference_budget import choose
from .trace_verifier import verify
from .phase2 import predictions
from .phase3 import OUT, REGISTER, compare, measure


def transfer():
    study = read(REGISTER)
    chosen = read(OUT / "selection.json")["selected"]
    retention3.build()
    prose = [json.loads(line) for line in (retention3.ROOT / "validation.jsonl").open()]
    v3.build()
    results = {}
    for split in ("development", "development_ood"):
        sample = v2.rows(split, "trace", v3.ROOT)
        m = {
            **study,
            "evaluation_dataset": read(v3.ROOT / "manifest.json"),
            "retention_manifest": read(retention3.ROOT / "manifest.json"),
            "generation": {**study["generation"], "split": split, "count": len(sample)},
        }
        scores = {}
        for arm in ("dense", "direct", chosen):
            for mode in ("trained",) if arm == "dense" else ("trained", "bypass"):
                scores[arm + "/" + mode] = measure(arm, 100000, mode, sample, prose, m)
        results[split] = dict(
            metrics={
                k: dict(
                    correct=r["certificate"]["correct"],
                    n=len(sample),
                    verified=r["certificate"]["verified"],
                    truncated=r["certificate"]["truncated"],
                    final_nll=r["likelihood"]["nll"]["final"],
                    retention=r["retention"]["nll"],
                    seconds=r["generations"]["seconds"],
                    eval_id=fingerprint(r["identity"]),
                )
                for k, r in scores.items()
            },
            paired={
                arm: compare(scores["dense/trained"], scores[arm + "/trained"])
                for arm in ("direct", chosen)
            },
            bypass={
                arm: compare(scores[arm + "/bypass"], scores[arm + "/trained"])
                for arm in ("direct", chosen)
            },
        )
        save(OUT / "fresh-development.json", results)


def strict_records(measurement, samples):
    for row, pred in zip(samples, measurement["records"], strict=True):
        status = verify(row, pred["text"], pred["token_cap_reached"])["final"]
        pred["correct"] = status == "correct"
        if status not in ("correct", "incorrect"):
            pred["prediction"] = None


def frontier():
    chosen = read(OUT / "selection.json")["selected"]
    docs = v2.rows("validation", "trace")[:60]
    # Two crossover halves. Each arm runs on identical prompts without sharing its KV history.
    work = {}
    batches = {}
    timing = {}
    for half, arms in enumerate(
        (("dense", "gated", "bypass"), ("bypass", "gated", "dense"))
    ):
        sample = docs[half * 30 : (half + 1) * 30]
        for arm in arms:
            root = Path("checkpoints") / (
                "p3_dense" if arm == "dense" else "p3_" + chosen
            )
            m, t, _ = checkpoint.load_trained(
                root / read(root / "targets-100000.json")["directory"]
            )
            if arm == "bypass":
                m.span = replace(m.span, passes=1)
            predictions(m, t, sample[:1], 8)
            for b in (1, 8):
                rec = predictions(
                    m, t, sample, 192, batch_size=b, padded=True, detailed=True
                )
                strict_records(rec, sample)
                batches.setdefault((arm, b), []).extend(rec["batches"])
                work.setdefault((arm, b), []).extend(rec["records"])
                timing[(arm, b)] = timing.get((arm, b), 0) + rec["seconds"]
            del m
            gc.collect()
            mx.clear_cache()
    root = Path("checkpoints/p3_dense")
    m, t, _ = checkpoint.load_trained(
        root / read(root / "targets-100000.json")["directory"]
    )
    used = timing["dense", 1]
    budget = timing["gated", 1]
    votes = []
    candidates = []
    for doc, base in zip(docs, work["dense", 1]):
        options = [base]
        if used < budget:
            for seed in (31001, 31002):
                r = predictions(
                    m,
                    t,
                    [doc],
                    192,
                    temperature=0.7,
                    seed=seed,
                    batch_size=1,
                    detailed=True,
                )
                strict_records(r, [doc])
                options.extend(r["records"])
                used += r["seconds"]
        votes.append(choose(options))
        candidates.append(options)
    save(
        OUT / "frontier.json",
        dict(
            n=len(docs),
            greedy={
                f"{a}/{b}": dict(
                    correct=sum(r["correct"] for r in rs),
                    seconds=timing[a, b],
                    records=rs,
                    batches=batches[a, b],
                )
                for (a, b), rs in work.items()
            },
            plurality=dict(
                correct=sum(v == r["gold"] for v, r in zip(votes, work["dense", 1])),
                seconds=used,
                budget=budget,
                votes=votes,
                candidates=candidates,
            ),
        ),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["transfer", "frontier"])
    args = p.parse_args()
    (transfer if args.task == "transfer" else frontier)()
