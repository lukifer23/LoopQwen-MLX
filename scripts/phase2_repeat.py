"""Repeat the same native run and quantify initialization-to-checkpoint divergence."""

from pathlib import Path

import mlx.core as mx
from phase2_configs import write
from phase2_register import register
from phase2_study import execute

from loopqwen.checkpoint import location
from loopqwen.evaluation.phase2 import evaluate
from loopqwen.io import evaluation_record, read, save

if __name__ == "__main__":
    run = write("repeat_direct", steps=24, eval_interval=24)
    execute(run)
    register(
        "phase2_repeat",
        [run.stem],
        "Replayed identical native seed and update schedule.",
    )
    evaluate("phase2_repeat")
    orig = location("checkpoints/p2_screen_direct")
    rep = location(Path("checkpoints") / run.stem)
    a, b = [mx.load(str(p / "adapters.safetensors")) for p in (orig, rep)]
    # B starts at zero. Measuring only these factors avoids diluting the error with initialized A.
    keys = [k for k in a if k.endswith("lora_b")]
    relative = mx.sqrt(
        sum(mx.sum((a[k] - b[k]) ** 2) for k in keys)
        / sum(mx.sum(a[k] ** 2) for k in keys)
    ).item()
    states = [read(p / "state.json") for p in (orig, rep)]
    assert states[0]["order_hash"] == states[1]["order_hash"]
    out = Path("outputs/phase2")
    scores = [
        read(evaluation_record(out / group, arm))["scores"]["validation"]
        for group, arm in (
            ("phase2_screen", "p2_screen_direct"),
            ("phase2_repeat", run.stem),
        )
    ]
    save(
        out / "repeat.json",
        dict(
            rank_factor_b_relative=relative,
            validation_nll=[s["trace"]["loss"]["nll"] for s in scores],
            answer_accuracy=[s["answer"]["generations"]["correct"] for s in scores],
            n=scores[0]["answer"]["generations"]["n"],
        ),
    )
