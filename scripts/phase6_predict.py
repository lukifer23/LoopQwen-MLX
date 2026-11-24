"""Use the published Phase 6 categorical state head at a fixed request depth."""

import argparse
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from loopqwen import checkpoint
from loopqwen.evaluation.mechanism import infer, unpack
from loopqwen.io import digest, read
from loopqwen.training.mechanism import StudyArm, assets, model_for, requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/phase6"))
    p.add_argument(
        "--arm", default="randomized", choices=["randomized", "canonical", "outcome"]
    )
    p.add_argument("--map", nargs=8, type=int, choices=range(8), metavar="SUCCESSOR")
    p.add_argument(
        "--order", nargs=8, type=int, default=list(range(8)), metavar="SOURCE"
    )
    p.add_argument("--start", type=int, choices=range(8), default=0)
    p.add_argument("--depth", type=int, choices=range(1, 7), default=3)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    study = read(args.archive / "results.json")
    arm = StudyArm.registered(
        "randomized" if args.arm == "outcome" else args.arm, study["seed"]
    )
    if args.arm == "outcome":
        arm = replace(arm, name="outcome", objective="outcome", milestones=(80,))
    folder = args.archive / arm.name
    info = study["arms"][arm.name]
    if digest(folder / "adapters.safetensors") != info["weights"]:
        raise ValueError("Checkpoint hash mismatch")
    assets()
    m, t, _ = model_for(arm)
    checkpoint.restore(folder, m)
    m.eval()
    if args.verify:
        r = unpack(args.archive / info["evaluations"][-1]["path"])["records"][:16]
        probs, _ = infer(m, arm, [requests(r, 1, arm, t, {})], 3, False)
        expected = mx.array([v["probabilities"] for v in r])
        error = mx.max(mx.abs(probs - expected)).item()
        if (
            error > 1e-5
            or not mx.array_equal(
                mx.argmax(probs, axis=-1), mx.argmax(expected, axis=-1)
            ).item()
        ):
            raise ValueError(error)
        print(arm.name, "48 predictions verified, max probability difference", error)
    else:
        if args.map is None or sorted(args.order) != list(range(8)):
            p.error("Supply eight successors and a valid presentation order")
        row = {"map": args.map, "order": args.order, "start": args.start}
        probs, _ = infer(m, arm, [requests([row], 1, arm, t, {})], args.depth, False)
        print(mx.argmax(probs, axis=-1).tolist()[0])


if __name__ == "__main__":
    main()
