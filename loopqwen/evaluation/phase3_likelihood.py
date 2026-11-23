"""Answer likelihood from the prompt alone and retention relative to the untouched base."""

import gc
from pathlib import Path

import mlx.core as mx

from .. import checkpoint
from ..data import v2, v3, retention3
from ..io import read, save
from ..looped_model import LoopedModel
from ..model_loader import load_base
from ..recurrence import Span
from ..training.losses import validation_nll
from .phase2_controls import representation


def run():
    from dataclasses import replace
    import json

    chosen = read("outputs/phase3/selection.json")["selected"]
    suites = {
        "v2": v2.rows("validation", "answer")[:60],
        "v3": v2.rows("development", "answer", v3.ROOT),
        "v3_ood": v2.rows("development_ood", "answer", v3.ROOT),
    }
    prose = [json.loads(line) for line in (retention3.ROOT / "validation.jsonl").open()]
    metrics = {}
    for arm in ("base", "dense", "direct", chosen):
        if arm == "base":
            native, _, _ = load_base()
            model = LoopedModel(native, Span(passes=1))
            del native
        else:
            root = Path("checkpoints") / ("p3_" + arm)
            model, _, _ = checkpoint.load_trained(
                root / read(root / "targets-100000.json")["directory"]
            )
        model.eval()
        for k in (1,) if arm in ("base", "dense") else (1, 2):
            model.span = replace(model.span, passes=k)
            r = dict(
                answer_only={
                    name: validation_nll(model, docs) for name, docs in suites.items()
                },
                v2_prose=validation_nll(
                    model, v2.rows("language_validation", "language")
                ),
                v3_retention=validation_nll(model, prose),
                domains={},
                representation=[
                    representation(model, mx.array([doc["ids"][: doc["prefix"]]]))
                    for doc in suites["v2"][:5]
                ],
            )
            for domain in sorted({p["category"] for p in prose}):
                r["domains"][domain] = validation_nll(
                    model, [p for p in prose if p["category"] == domain]
                )
            metrics[f"{arm}/{k}"] = r
        del model
        gc.collect()
        mx.clear_cache()
    save("outputs/phase3/answer-likelihood.json", metrics)


if __name__ == "__main__":
    run()
