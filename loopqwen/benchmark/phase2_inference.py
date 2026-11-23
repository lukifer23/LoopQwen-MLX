"""Depth diagnostics and completed prefill/decode time at extended prompt lengths."""

import statistics
from dataclasses import replace
from time import perf_counter

import mlx.core as mx

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..evaluation.phase2 import nll, predictions
from ..io import save


def measured(model, ids):
    cache = model.make_cache()
    start = perf_counter()
    x = mx.array([ids])
    logits = model.project(model.hidden(x, cache)[:, -1])
    value = mx.argmax(logits, -1)
    mx.eval(value)
    first = perf_counter()
    # Fixed-work latency test. Consume exactly 12 incremental steps even if EOS is emitted.
    for step in range(12):
        logits = model.project(model.hidden(value[:, None], cache)[:, -1])
        value = mx.argmax(logits, -1)
        mx.eval(value)
    return first - start, (perf_counter() - first) / 12


def run():
    model, tok, state = load_trained("checkpoints/p2_confirm_43_loop")
    samples = rows("validation", "answer")
    tokens = sum([r["ids"][: r["prefix"]] for r in samples[:40]], [])
    result = {}
    for k in (1, 2, 3):
        model.span = replace(model.span, passes=k)
        latencies = []
        for length in (128, 512, 1024):
            mx.reset_peak_memory()
            measured(model, tokens[:length])
            times = [measured(model, tokens[:length]) for _ in range(3)]
            pre, dec = map(statistics.median, zip(*times))
            latencies.append(
                dict(
                    length=length,
                    samples=times,
                    median_prefill_tps=length / pre,
                    median_decode_tps=1 / dec,
                    peak=mx.get_peak_memory(),
                )
            )
        result[k] = dict(
            development_loss=nll(model, samples[:90]),
            greedy=predictions(model, tok, samples[:30], 12),
            timing=latencies,
            executions=len(model.sites),
        )
    save("outputs/phase2/depth_latency.json", result)


if __name__ == "__main__":
    run()
