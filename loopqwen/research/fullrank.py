"""Fit the full-rank internal post-training update on disjoint calibration text, then train K2."""

import argparse
import gc
import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_tokenizer

from .. import backbone_mix, checkpoint
from ..deployment import load_trained
from ..evaluation.language7 import base
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from ..training.losses import validation_nll
from . import all_layer, bootstrap, core, projected_distillation, rationale_transfer

NAME = "fullrank-reasoning-01"
REG = Path("research/experiments") / (NAME + ".json")
PANEL = Path("research/data") / (NAME + ".json")
TRAIN = PANEL.with_name("fullrank-reasoning-training.json")


def material(s):
    source, rows, manifest = all_layer.material(s)
    t = load_tokenizer(local_snapshot())
    anchors = projected_distillation.anchors(
        t, dict(s, updates=s["updates"] + s["calibration_docs"])
    )
    raw = [{"ids": d["ids"], "prefix": 1} for d in anchors]
    calibration = [dict(d, input_length=len(d["ids"]) - 1) for d in raw[s["updates"] :]]
    training = []
    for j, d in enumerate(raw[: s["updates"]]):
        training.extend([*source[4 * j : 4 * j + 4], d])
    summary = dict(
        manifest,
        documents=fingerprint(training),
        raw_training=fingerprint(raw[: s["updates"]]),
        calibration=fingerprint(calibration),
        response_tokens=sum(len(d["ids"]) - d["prefix"] for d in training),
    )
    return rows, training, calibration, summary


def prepare(s, samples, budget, root):
    control, _, _ = base()
    reference = validation_nll(control, samples, batch_size=1, projection_chunk=32)[
        "nll"
    ]
    del control
    gc.collect()
    mx.clear_cache()
    config = dict(
        read(Path(s["parent"]) / "state.json")["spec"]["deployment"]["config"],
        seed=s["seed"],
        target="all",
        gradient_checkpointing=True,
        passes=s["depth"],
    )
    m, _, _ = checkpoint.setup(config)
    m.eval()
    donor, _ = projected_distillation.tutor(s)
    original = backbone_mix.layer_weights(m)
    table = []
    for fraction in s["fractions"]:
        budget.check()
        count = backbone_mix.apply(m, original, donor, fraction)
        nll = validation_nll(m, samples, batch_size=1, projection_chunk=32)["nll"]
        table.append({"fraction": fraction, "nll": nll, "tensors": count})
    eligible = [v for v in table if v["nll"] <= reference + s["calibration_tolerance"]]
    chosen = max(eligible, key=lambda v: v["fraction"]) if eligible else None
    save(
        root / "calibration.json",
        {"base": reference, "rows": table, "selected": chosen},
    )
    if not chosen:
        return None, None
    backbone_mix.apply(m, original, donor, chosen["fraction"])
    dep = {
        "config": config,
        "alpha_override": None,
        "backbone_mix": {
            "donor": s["teacher"],
            "fraction": chosen["fraction"],
            "tensors": chosen["tensors"],
        },
    }
    return m, dep


def run(freeze):
    s = read(REG)
    rows, docs, calibration, manifest = material(s)
    for path, obj in ((PANEL, rows), (TRAIN, manifest)):
        if freeze and not path.exists():
            save(path, obj)
        if read(path) != obj:
            raise ValueError("Fullrank study material changed")
    if freeze:
        return
    registry = read("research/registry.json")
    if registry["experiments"][NAME]["registration"] != digest(REG) or any(
        registry["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ):
        raise ValueError("Missing frozen registration")
    if any(
        digest(Path(s[k]) / "adapters.safetensors") != s[h]
        for k, h in (("parent", "parent_sha256"), ("dense_control", "dense_sha256"))
    ):
        raise ValueError("Historical checkpoint identity changed")
    root = core.OUT / NAME
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    b = core.Budget(NAME, s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
    }
    try:
        model, dep = prepare(s, calibration, b, root)
        if model is None:
            result.update(status="no_retained_mixture")
        else:
            gc.collect()
            mx.clear_cache()
            cp = Path("checkpoints/research") / NAME
            parent = fingerprint([s["base_revision"], dep["backbone_mix"]])
            state = core.train(model, docs, dict(s, deployment=dep), cp, b, parent)
            gen = checkpoint_location(cp)
            bootstrap.release(gen, root / "model")
            w = digest(gen / "adapters.safetensors")
            result.update(
                weights=w,
                updates=state["step"],
                targets=sum(h["targets"] for h in state["history"]),
                backbone_mix=dep["backbone_mix"],
            )
            probe_text = "The notebook contains the observations from yesterday."
            tokenizer = load_tokenizer(local_snapshot())
            probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)
            probe_input = mx.array([probe_ids])
            reference = np.asarray(model(probe_input)[:, -1].astype(mx.float32))
            del model
            gc.collect()
            mx.clear_cache()
            loaded, _, _ = load_trained(root / "model")
            restored = np.asarray(loaded(probe_input)[:, -1].astype(mx.float32))
            if not np.array_equal(reference, restored):
                raise ValueError(
                    "Reloaded full-rank decoder differs from the trained endpoint"
                )
            save(
                root / "reload-parity.json",
                {
                    "prompt": probe_text,
                    "ids": probe_ids,
                    "vocabulary": reference.size,
                    "logits_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                },
            )
            del loaded
            gc.collect()
            mx.clear_cache()
            rationale_transfer.evaluate_pair(
                root, root / "model", rows, s, w, b, result, retention=True
            )
            a = result["full"]
            result["passed"] = (
                a["all"]["wins"] - a["all"]["losses"] >= s["net_floor"]
                and all(
                    a[k]["wins"] > a[k]["losses"]
                    and result["prefix_256"][k]["wins"]
                    >= result["prefix_256"][k]["losses"]
                    for k in s["sources"]
                )
                and a["all"]["candidate"]["seconds"] <= a["all"]["base"]["seconds"]
                and result["retention"]["candidate"]["nll"]
                <= result["retention"]["base"]["nll"] + 0.03
            )
    except (InterruptedError, TimeoutError, MemoryError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    run(parser.parse_args().freeze)
