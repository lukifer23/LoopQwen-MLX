"""Independent uncached, batch-one terminal readout latency beyond trained depths."""

from statistics import median
from time import perf_counter
import mlx.core as mx
from ..data import pointer5, pointer6
from ..io import save, read
from ..training.exposure import ARM, OUT, PARENT
from ..training.mechanism import requests
from ..training.pointer5 import inputs, readout
from .mechanism import load


def predict(model, depth, x, last):
    h = model.prefix_hidden(x)
    for _ in range(depth):
        h = model.shared_block(h)
    mx.eval(mx.argmax(readout(model, h, last), axis=-1))


def run():
    n = read(OUT / "decision.json")["step"]
    result = {}
    for a, step in [(PARENT, 480), (ARM, n)]:
        m, t, _, w = load(a, step)
        rows = [
            pointer6.ordered(x, "random0", 61091)
            for x in pointer5.rows("development")[:16]
        ]
        batches = [inputs(requests([x], 1, a, t, {})) for x in rows]
        mx.eval(batches)
        cost = {}
        for k in (4, 5, 6):

            predict(m, k, *batches[0])
            samples = []
            for batch in batches:
                start = perf_counter()
                predict(m, k, *batch)
                samples.append(perf_counter() - start)
            cost[k] = dict(median_ms=1000 * median(samples), seconds=samples)
        result[step] = dict(
            weights=w, latency=cost, ids=[r["example_id"] for r in rows]
        )
        del m
    save(OUT / "long-latency.json", result)
