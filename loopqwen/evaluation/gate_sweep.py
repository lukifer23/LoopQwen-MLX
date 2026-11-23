"""Sensitivity of a trained interpolation gate on development examples."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..io import save
from .phase2 import nll
from .phase2_controls import representation


def run():
    model, tok, state = load_trained(Path("checkpoints/p2_screen_stabilized"))
    answers = rows("validation", "answer")[:90]
    traces = rows("validation", "trace")[:90]
    learned = mx.sigmoid(model.alpha_logit).item()
    result = []
    for depth, alpha in [(1, None), (2, 0.0), (2, None), (2, 0.5), (2, 1.0), (3, None)]:
        model.span = replace(model.span, passes=depth)
        model.alpha_override = alpha
        result.append(
            dict(
                k=depth,
                alpha=learned if alpha is None else alpha,
                answer=nll(model, answers),
                trace=nll(model, traces),
                norms=[
                    representation(model, mx.array([r["ids"][: r["prefix"]]]))
                    for r in answers[:5]
                ],
            )
        )
    save("outputs/phase2/gate_sweep.json", result)


if __name__ == "__main__":
    run()
