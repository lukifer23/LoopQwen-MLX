"""Load a published categorical adapter and state head over the pinned Qwen Base."""

import argparse
import gzip
import json
from pathlib import Path

import mlx.core as mx
from loopqwen import checkpoint
from loopqwen.io import digest, read
from loopqwen.training.pointer5 import architecture, inputs, outputs, request_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("docs/measurements/phase5")
    )
    parser.add_argument("--arm", default="recurrent")
    parser.add_argument("--map", nargs=8, type=int, metavar="NODE")
    parser.add_argument("--start", type=int, choices=range(8), default=0)
    parser.add_argument("--depth", type=int, choices=range(1, 4), default=3)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare with eight archived requests at each trained depth",
    )
    args = parser.parse_args()
    index = read(args.archive / "training.json")
    if args.arm not in index or not (args.archive / "models" / args.arm).is_dir():
        parser.error("Choose an archived categorical arm with published weights")
    entry = index[args.arm]
    gen = args.archive / "models" / args.arm
    if digest(gen / "adapters.safetensors") != entry["weights"]:
        raise ValueError("Published categorical checkpoint hash mismatch")
    meta = entry["provenance"]
    model, tok, _, _ = architecture(meta["seed"], args.arm, "off")
    checkpoint.restore(gen, model)
    model.eval()
    if args.verify:
        with gzip.open(
            args.archive / f"predictions/{args.arm}-{meta['step']}.json.gz", "rt"
        ) as f:
            rows = json.load(f)["records"][:8]
        x, last = inputs(request_rows(rows, 1, args.arm, tok, {}))
        for j in (1, 2, 3):
            probs = mx.softmax(
                outputs(model, x, last, j, args.arm)[-1].astype(mx.float32), axis=-1
            )
            reference = mx.array([r["probabilities"][j - 1] for r in rows])
            err = mx.max(mx.abs(probs - reference)).item()
            hits = (
                (mx.argmax(probs, axis=1) == mx.argmax(reference, axis=1)).sum().item()
            )
            print(
                f"{args.arm} pass {j}: {hits}/8 archived predictions, maximum probability difference {err:.6g}"
            )
            if hits != len(rows) or err > 1e-5:
                raise ValueError("Fresh-process readout differs")
    else:
        if args.map is None or sorted(args.map) != list(range(8)):
            parser.error("--map requires the eight unique successors in node order")
        row = dict(map=args.map, start=args.start, order=list(range(8)))
        x, last = inputs(request_rows([row], args.depth, args.arm, tok, {}))
        scores = outputs(model, x, last, args.depth, args.arm)
        print(
            json.dumps(
                [
                    dict(
                        hop=j,
                        state=mx.argmax(z).item(),
                        probabilities=mx.softmax(
                            z.astype(mx.float32), axis=-1
                        ).tolist()[0],
                    )
                    for j, z in enumerate(scores, 1)
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
