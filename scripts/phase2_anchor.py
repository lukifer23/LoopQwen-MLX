"""Development-only input re-injection versus the completed direct and interpolation controls."""

from dataclasses import replace
from pathlib import Path
import mlx.core as mx
from phase2_configs import write
from phase2_study import execute
from phase2_register import register
from loopqwen.checkpoint import load_trained
from loopqwen.data.v2 import rows
from loopqwen.evaluation.phase2 import evaluate
from loopqwen.io import save


def run():
    p = write(
        "anchor",
        steps=24,
        eval_interval=24,
        mode="input_anchor",
        alpha_init=0.9,
        validation_batch_size=4,
    )
    execute(p)
    model, tok, state = load_trained(Path("checkpoints") / p.stem)
    x = mx.array([rows("validation", "answer")[0]["ids"][:35]])
    span = model.span
    model.span = replace(span, passes=1)
    single = model(x)
    model.span = replace(span, mode="direct", passes=1)
    if not mx.array_equal(single, model(x)).item():
        raise AssertionError("K=1 input mixing altered logits")
    model.span = span
    cache = model.make_cache()
    uncached = model(x)
    cached = mx.concatenate(
        [model(x[:, i : i + 1], cache) for i in range(x.shape[1])], axis=1
    )
    rms = mx.sqrt(
        mx.mean((cached.astype(mx.float32) - uncached.astype(mx.float32)) ** 2)
        / mx.mean(uncached.astype(mx.float32) ** 2)
    ).item()
    if rms > 0.03:
        raise AssertionError(rms)
    save(
        "outputs/phase2/anchor_diagnostics.json",
        dict(
            k1_exact=True,
            cache_relative_rms=rms,
            initial_alpha=0.9,
            final_alpha=state["history"][-1]["alpha"],
        ),
    )
    # Independent cohort, no feedback into the frozen confirmatory comparison.
    register(
        "phase2_anchor",
        [p.stem, "p2_screen_direct", "p2_screen_stabilized"],
        "Development-only block-input re-injection versus direct and output interpolation.",
    )
    evaluate("phase2_anchor")


if __name__ == "__main__":
    run()
