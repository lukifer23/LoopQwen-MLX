"""Synchronized fixed-work state prediction timings on the first 32 development maps."""

from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx

from ..data.pointer5 import rows
from ..io import digest, read, save
from ..training.pointer5 import OUT, SPEC, inputs, outputs, request_rows, root_for
from .pointer5_diagnostics import load


def measure(seed):
    cohort = rows("development")[:32]
    arms = list(SPEC["arms"])
    if (root_for(seed, "bridge") / "at-480.json").is_file():
        arms.append("bridge")
    for arm in arms:
        m, tok, weight = load(seed, arm, 480)
        path = OUT / str(seed) / f"latency-{arm}.json"
        identity = dict(
            weights=weight,
            implementation=digest(Path(__file__)),
            items=[r["example_id"] for r in cohort],
        )
        if path.exists():
            if read(path)["identity"] != identity:
                raise ValueError("Incompatible previous categorical timing report")
            continue
        results = []
        cache = {}
        for n in (1, 2, 3):
            requests = request_rows(cohort, n, arm, tok, cache)
            for size in (1, 8):
                batches = [
                    inputs(requests[i : i + size]) for i in range(0, len(cohort), size)
                ]
                for x, last in batches:
                    mx.eval(x, last)

                def predict(batch):
                    x, last = batch
                    pred = mx.argmax(outputs(m, x, last, n, arm)[-1], axis=-1)
                    mx.eval(pred)

                predict(batches[0])
                times = []
                for batch in batches:
                    start = perf_counter()
                    predict(batch)
                    times.append(perf_counter() - start)
                results.append(
                    dict(
                        depth=n,
                        batch=size,
                        seconds=times,
                        median_batch_seconds=median(times),
                        maps_per_second=len(cohort) / sum(times),
                    )
                )
        save(path, dict(identity=identity, results=results))
