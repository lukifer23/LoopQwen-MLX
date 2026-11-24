"""Reload the compact Phase 7 checkpoint over the pinned Qwen weights."""

import argparse
from pathlib import Path
import mlx.core as mx
from loopqwen.training.exposure import ARM
from loopqwen.training.mechanism import model_for, requests, assets
from loopqwen import checkpoint
from loopqwen.io import digest, read
from loopqwen.evaluation.mechanism import infer, unpack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/phase7"))
    p.add_argument("--verify", action="store_true")
    p.add_argument("--map", type=int, nargs=8)
    p.add_argument("--start", type=int, choices=range(8), default=0)
    p.add_argument("--depth", type=int, choices=range(1, 7), default=3)
    a = p.parse_args()
    root = a.archive
    if (
        digest(root / "model/adapters.safetensors")
        != read(root / "checksums.json")["model/adapters.safetensors"]
    ):
        raise ValueError("Changed release")
    assets()
    m, t, _ = model_for(ARM)
    checkpoint.restore(root / "model", m)
    m.eval()
    if a.verify:
        rows = unpack(root / "endpoint.json.gz")["records"][:16]
        prob, _ = infer(m, ARM, [requests(rows, 3, ARM, t, {})], 3, False)
        expected = mx.array([r["probabilities"] for r in rows])
        error = mx.max(mx.abs(prob - expected)).item()
        if (
            error > 1e-5
            or not mx.array_equal(prob.argmax(-1), expected.argmax(-1)).item()
        ):
            raise ValueError(error)
        print("48 argmax readouts verified; max probability difference", error)
    else:
        if a.map is None or sorted(a.map) != list(range(8)):
            p.error("Supply eight unique successors")
        r = dict(map=a.map, start=a.start, order=list(range(8)))
        prob, _ = infer(m, ARM, [requests([r], a.depth, ARM, t, {})], a.depth, False)
        print(prob.argmax(-1).tolist()[0])


if __name__ == "__main__":
    main()
