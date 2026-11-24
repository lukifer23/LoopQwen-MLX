"""Reload a published RSI generation and compute its categorical state trajectory."""

import argparse
from pathlib import Path

import mlx.core as mx
from loopqwen import checkpoint
from loopqwen.io import digest, read
from loopqwen.training.mechanism import model_for, assets, requests
from loopqwen.training.exposure import ARM
from loopqwen.evaluation.mechanism import infer, unpack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/rsi1"))
    p.add_argument(
        "--arm",
        default="adaptive2",
        choices=("static1", "adaptive1", "static2", "adaptive2", "time"),
    )
    p.add_argument("--map", type=int, nargs=8)
    p.add_argument("--start", type=int, choices=range(8), default=0)
    p.add_argument("--depth", type=int, choices=range(1, 7), default=6)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.arm == "time":
        root = args.archive / "model"
        data = args.archive / "evaluation.json.gz"
        expected = read(args.archive / "decision.json")["weights"]
    else:
        root = args.archive / "models" / args.arm
        data = args.archive / (args.arm + "-evaluation.json.gz")
        expected = read(args.archive / (args.arm + "-lineage.json"))["weights"]
    if digest(root / "adapters.safetensors") != expected:
        raise ValueError("Wrong checkpoint")
    assets()
    m, t, _ = model_for(ARM)
    checkpoint.restore(root, m)
    m.eval()
    if args.verify:
        rec = unpack(data)
        if rec["weights"] != expected:
            raise ValueError("Wrong evaluation")
        rows = rec["records"][:8]
        depth = 6
    else:
        if args.map is None or sorted(args.map) != list(range(8)):
            p.error("Supply eight unique successors")
        rows = [dict(map=args.map, order=list(range(8)), start=args.start)]
        depth = args.depth
    pred, _ = infer(m, ARM, [requests(rows, depth, ARM, t, {})], depth, False)
    if args.verify:
        ref = mx.array([r["probabilities"] for r in rows])
        error = mx.max(mx.abs(ref - pred)).item()
        if error > 1e-5 or not mx.array_equal(pred.argmax(-1), ref.argmax(-1)).item():
            raise ValueError(error)
        print(
            args.arm, "48 native readouts verified, max probability difference", error
        )
    else:
        print(pred.argmax(-1).tolist()[0])


if __name__ == "__main__":
    main()
