"""Time real V2 validation and bound the temporary vocabulary projection."""

from statistics import median
from time import perf_counter

import mlx.core as mx

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..io import save
from ..training.losses import validation_nll


def run():
    model, tok, _ = load_trained("checkpoints/p2_confirm_43_loop")
    out = {}
    for form in ("answer", "trace"):
        samples = rows("validation", form)[:30]
        runs = []
        for size, chunk in ((1, None), (4, None), (4, 96)):
            times = []
            peaks = []
            for i in range(3):
                mx.clear_cache()
                mx.reset_peak_memory()
                mx.synchronize()
                start = perf_counter()
                metric = validation_nll(model, samples, size, chunk)
                mx.synchronize()
                times.append(perf_counter() - start)
                peaks.append(mx.get_peak_memory())
            runs.append(
                dict(
                    batch=size,
                    projection_chunk=chunk,
                    times=times[1:],
                    seconds=median(times[1:]),
                    peak=max(peaks[1:]),
                    **metric,
                )
            )
        for row in runs:
            if (
                row["tokens"] != runs[0]["tokens"]
                or abs(row["nll"] - runs[0]["nll"]) > 0.005
            ):
                raise AssertionError((form, runs))
        out[form] = runs
    save("outputs/phase2/validation_memory.json", out)


if __name__ == "__main__":
    run()
