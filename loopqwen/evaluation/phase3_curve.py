"""Re-score immutable milestone generations and exact initialized models on the same development cohort."""

import gc
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from .. import checkpoint
from ..config import config
from ..data.v2 import rows
from ..io import read, save
from ..training.losses import validation_nll


def run():
    selected = read("outputs/phase3/selection.json")["selected"]
    docs = {
        protocol: rows("validation", protocol)[:30] for protocol in ("trace", "answer")
    }
    prose = rows("language_validation", "language")[:24]
    results = []
    screened = read("outputs/phase3/selection.json")["evidence"]
    for arm in dict.fromkeys(row["arm"] for row in screened):
        root = Path("checkpoints") / ("p3_" + arm)
        c = config("outputs/phase3/configs/p3_" + arm + ".yaml")
        for target in (0, 10000, 25000, 50000, 100000):
            if arm not in ("dense", "direct", selected) and target > 25000:
                continue
            if target:
                pointer = root / f"targets-{target}.json"
                gen = root / read(pointer)["directory"]
                model, _, state = checkpoint.load_trained(gen)
                step = state["step"]
                targets = state["loss_tokens"]
            else:
                model, _, _ = checkpoint.setup(c)
                model.eval()
                step = targets = 0
            for k in (1,) if arm == "dense" else (1, 2):
                model.span = replace(model.span, passes=k)
                metrics = {
                    p: validation_nll(model, sample) for p, sample in docs.items()
                }
                results.append(
                    dict(
                        arm=arm,
                        step=step,
                        targets=targets,
                        k=k,
                        alpha=float(model.effective_alpha())
                        if model.span.has_gate and k > 1
                        else None,
                        retention=validation_nll(model, prose),
                        **metrics,
                    )
                )
            del model
            gc.collect()
            mx.clear_cache()
    save("outputs/phase3/learning-curves.json", results)


if __name__ == "__main__":
    run()
