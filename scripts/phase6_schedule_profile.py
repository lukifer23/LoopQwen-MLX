"""Time exact logical-batch construction with and without epoch-scoped reuse."""

import argparse
from pathlib import Path
from time import perf_counter

from loopqwen.data import pointer5, pointer6
from loopqwen.io import save
from loopqwen.training.pointer5 import schedule


def profile(seed):
    pointer6.build()
    rows = pointer5.rows("train")
    steps = len(rows) // pointer6.SPEC["batch"]
    times = {"old": [], "epoch_cached": []}
    for _ in range(5):
        start = perf_counter()
        reference = [schedule(seed, j, rows) for j in range(steps)]
        times["old"].append(perf_counter() - start)
        start = perf_counter()
        cached = [
            (pointer6.depth(j, 0), batch)
            for j, batch in enumerate(pointer6.epoch_batches(rows, seed, 0))
        ]
        times["epoch_cached"].append(perf_counter() - start)
        if cached != reference:
            raise ValueError("Changed example order or curriculum")
    return times


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose an unused output path")
    save(args.output, profile(args.seed))
