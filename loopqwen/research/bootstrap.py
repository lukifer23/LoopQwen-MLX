"""Compare complete verified language responses from a fixed parent and its SFT child."""

import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from . import core
from ..data import pointer5, pointer6, retention3
from ..evaluation.mechanism import infer, packed, unpack
from ..training.losses import validation_nll
from ..training.mechanism import requests
from ..training.verified_trace import request, verdict
from ..io import read, save, digest, checkpoint_location
from ..identity import fingerprint


def traces(m, t, rows, spec, budget, path, weight):
    rec = []
    cap = spec.get("generation_cap", 24)
    for r in rows:
        ids = t.encode(request(r), add_special_tokens=False)
        out = core.generate(m, t, ids, cap, budget.check, line=True)
        visible = out["visible"].strip()
        score = verdict(r["map"], r["start"], visible, out["stop"])
        rec.append(
            dict(
                id=r["example_id"],
                map_id=r["map_id"],
                map=r["map"],
                start=r["start"],
                order=r["order"],
                **out,
                score=score,
                correct=out["complete"] and score["final_correct"],
            )
        )
    packed(path, dict(weights=weight, records=rec))
    return dict(
        n=len(rec),
        verified=sum(r["score"]["verified"] for r in rec),
        correct=sum(r["correct"] for r in rec),
        completed=sum(r["complete"] for r in rec),
        tokens=sum(len(r["token_ids"]) for r in rec),
        seconds=sum(r["seconds"] for r in rec),
        median_ms=1000 * median(r["seconds"] for r in rec),
    )


def categorical(m, t, budget):
    from ..training.exposure import ARM

    pointer5.build()
    rows = [
        pointer6.ordered(r, "random0", 61091) for r in pointer5.rows("development")[:96]
    ]
    hit = np.zeros(3, dtype=int)
    for i in range(0, len(rows), 8):
        budget.check()
        g = rows[i : i + 8]
        p, _ = infer(m, ARM, [requests(g, 3, ARM, t, {})], 3, False)
        hit += (np.array(p.argmax(-1)) == np.array([r["states"][:3] for r in g])).sum(0)
    return hit.tolist()


def retention(m, k, budget):
    manifest = retention3.build()
    rows = [
        json.loads(l)
        for l in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    m.span = replace(m.span, passes=k)
    by = {}
    for domain in sorted({q["category"] for q in rows}):
        part = sorted(
            [q for q in rows if q["category"] == domain],
            key=lambda x: x["input_length"],
        )
        n = 0
        loss = 0
        for i in range(0, len(part), 2):
            budget.check()
            r = validation_nll(m, part[i : i + 2], batch_size=2, projection_chunk=32)
            n += r["tokens"]
            loss += r["nll"] * r["tokens"]
        by[domain] = dict(loss=loss, tokens=n, nll=loss / n)
    return dict(
        nll=sum(v["loss"] for v in by.values()) / sum(v["tokens"] for v in by.values()),
        by_domain=by,
        sha256=manifest["sha256"],
    )


def release(gen, folder):
    folder.mkdir(parents=True, exist_ok=False)
    for f in ("adapters.safetensors", "state.json"):
        shutil.copy(gen / f, folder / f)
    save(
        folder / "recovery.json",
        {
            "generation": str(gen),
            "optimizer_sha256": digest(gen / "optimizer.safetensors"),
            "weights": digest(gen / "adapters.safetensors"),
        },
    )


def run(name="trace-bootstrap-01"):
    spec = read(Path("research/experiments") / (name + ".json"))
    name = spec["id"]
    out = core.OUT / name
    if (out / "decision.json").exists():
        return print(read(out / "decision.json")["status"])
    budget = core.Budget(name, spec["limits"])
    try:
        rows = core.reserve(
            spec.get("dataset", name),
            spec["seed"],
            dict(train=spec["train_maps"], dev=spec["dev_maps"]),
        )
        parent = Path(spec["parent_path"])
        w = digest(parent / "adapters.safetensors")
        if w != spec["parent_sha256"]:
            raise ValueError("Parent mismatch")
        m, t = core.model(parent, spec["depth"])
        docs = [
            core.doc(t, request(r), " " + ", ".join(map(str, r["states"][:3])))
            for r in rows["train"]
        ]
        base = traces(m, t, rows["dev"], spec, budget, out / "parent.json.gz", w)
        cat0 = categorical(m, t, budget)
        print("parent", base, "categorical", cat0, flush=True)
        cp = Path("checkpoints/research") / name
        if name == "suffix-trace-01":
            core.suffix_adapters(m)
        from mlx.utils import tree_flatten

        capacity = sum(v.size for _, v in tree_flatten(m.trainable_parameters()))
        state_docs = None
        if spec.get("state_weight"):
            from ..training.exposure import ARM

            state_docs = [
                dict(ids=d["ids"], states=d["row"]["states"][:3])
                for d in requests(rows["train"], 3, ARM, t, {})
            ]
        st = core.train(m, docs, spec, cp, budget, w, state_docs=state_docs)
        gen = checkpoint_location(cp)
        child = digest(gen / "adapters.safetensors")
        trial = traces(m, t, rows["dev"], spec, budget, out / "child.json.gz", child)
        print("child", trial, flush=True)
        cat = categorical(m, t, budget)
        nll = {str(k): retention(m, k, budget) for k in (1, 3)}
        original = read("docs/measurements/phase7/transfer/decision.json")["retention"]
        n = len(rows["dev"])
        g = spec["gate"]
        checks = dict(
            verified=trial["verified"] / n >= g["verified_min"],
            gain=(trial["verified"] - base["verified"]) / n >= g["verified_gain"],
            categorical=(cat0[-1] - cat[-1]) / 96 <= g["max_hop3_drop"],
            retention=all(
                nll[str(k)]["nll"] - original[f"transfer-60/K{k}"]["nll"]
                <= g["max_nll_increase_each_depth"]
                for k in (1, 3)
            ),
        )
        if not (out / "model").exists():
            release(gen, out / "model")
        if name == "suffix-trace-01":
            save(
                out / "model/composition.json",
                dict(
                    frozen_parent=str(parent),
                    sha256=w,
                    trainable=capacity,
                    policy="last8-rank8",
                ),
            )
        save(
            out / "decision.json",
            dict(
                status="completed",
                parent=base,
                child=trial,
                checks=checks,
                promoted=all(checks.values()),
                categorical_before=cat0,
                categorical_after=cat,
                retention=nll,
                updates=st["step"],
                weights=child,
                trainable=capacity,
                target_tokens=sum(r["targets"] for r in st["history"]),
                training_seconds=sum(r["seconds"] for r in st["history"]),
            ),
        )
        print("checks", checks, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    run(*sys.argv[1:])
