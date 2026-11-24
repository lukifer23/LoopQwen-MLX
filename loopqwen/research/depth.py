"""Prospective same-task depth attribution on fresh explicit three-hop requests."""

import gc
from pathlib import Path
import mlx.core as mx
import numpy as np
from . import core, bootstrap
from ..evaluation.language7 import base
from ..evaluation.mechanism import unpack
from ..io import read, save, digest, checkpoint_location
from ..training.verified_trace import request


def run():
    spec = read("research/experiments/trace-depth-01.json")
    name = spec["id"]
    out = core.OUT / name
    if core.archived_result(out, spec):
        return
    budget = core.Budget(name, spec["limits"])
    try:
        cohort = core.reserve(name, spec["seed"], {"dev": spec["dev_maps"]})["dev"]
        training = core.reserve(
            spec["training_dataset"], spec["training_seed"], {"train": 256, "dev": 64}
        )["train"]
        parent = Path(spec["parent_path"])
        recurrent = Path(spec["recurrent_path"])
        for path, key in [(parent, "parent_sha256"), (recurrent, "recurrent_sha256")]:
            if digest(path / "adapters.safetensors") != spec[key]:
                raise ValueError("Registered weights changed")
        scores = {}
        for label in ("base", "dense", "bypass", "recurrent"):
            if label == "base":
                m, t, weight = base()
            elif label == "dense":
                m, t = core.model(parent, 1)
                docs = [
                    core.doc(t, request(r), " " + ", ".join(map(str, r["states"][:3])))
                    for r in training
                ]
                cp = Path("checkpoints/research") / name
                state = core.train(m, docs, spec, cp, budget, spec["parent_sha256"])
                gen = checkpoint_location(cp)
                weight = digest(gen / "adapters.safetensors")
                if not (out / "model").exists():
                    bootstrap.release(gen, out / "model")
            else:
                m, t = core.model(recurrent, 3 if label == "recurrent" else 1)
                weight = spec["recurrent_sha256"]
            scores[label] = bootstrap.traces(
                m, t, cohort, spec, budget, out / (label + ".json.gz"), weight
            )
            print(label, scores[label], flush=True)
            del m, t
            gc.collect()
            mx.clear_cache()
        records = {k: unpack(out / (k + ".json.gz"))["records"] for k in scores}
        keys = lambda rows: [(r["map_id"], r["prompt_ids"]) for r in rows]
        if any(keys(v) != keys(records["base"]) for v in records.values()):
            raise ValueError("Unmatched prompts")
        paired = {}
        draws = np.random.default_rng(spec["seed"]).integers(
            0, len(cohort), (5000, len(cohort))
        )
        for k in ("base", "dense", "bypass"):
            d = np.array(
                [
                    int(r["score"]["verified"]) - int(b["score"]["verified"])
                    for r, b in zip(records["recurrent"], records[k], strict=True)
                ]
            )
            paired[k] = dict(
                wins=int((d > 0).sum()),
                losses=int((d < 0).sum()),
                delta=float(d.mean()),
                ci=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist(),
            )
        save(
            out / "decision.json",
            dict(
                scores=scores,
                paired=paired,
                passed=all(paired[k]["delta"] >= 0.05 for k in ("bypass", "dense")),
                training_seconds=sum(r["seconds"] for r in state["history"]),
            ),
        )
    finally:
        budget.close()


if __name__ == "__main__":
    run()
