"""Generate or reproduce a complete trace using a published research adapter."""

import argparse
from pathlib import Path
from time import monotonic

from loopqwen.research import core
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, digest
from loopqwen.training.verified_trace import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(1, 3), default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--map", type=int, nargs=8, choices=range(8))
    parser.add_argument("--start", type=int, choices=range(8), default=0)
    args = parser.parse_args()
    root = args.archive
    key = "model/adapters.safetensors"
    if digest(root / key) != read(root / "checksums.json")[key]:
        raise ValueError("Published checkpoint changed")
    m, t = core.model(root / "model", args.depth)
    deadline = monotonic() + 60

    def guard():
        if monotonic() > deadline:
            raise TimeoutError("Prediction budget")

    if args.verify:
        name = "dense" if (root / "dense.json.gz").is_file() else "child"
        expected_depth = 1 if name == "dense" else 3
        if args.depth != expected_depth:
            parser.error(f"The archived outputs use K={expected_depth}")
        for r in unpack(root / (name + ".json.gz"))["records"][:8]:
            out = core.generate(m, t, r["prompt_ids"], 24, guard, line=True)
            if out["token_ids"] != r["token_ids"] or out["stop"] != r["stop"]:
                raise ValueError("Native generation differs from archive")
        print("Eight native complete token streams match the release")
    else:
        if args.map is None:
            parser.error("Supply --map or --verify")
        ids = t.encode(
            request(dict(map=args.map, order=list(range(8)), start=args.start)),
            add_special_tokens=False,
        )
        out = core.generate(m, t, ids, 24, guard, line=True)
        print(out["visible"].strip(), out["stop"], len(out["token_ids"]), "tokens")


if __name__ == "__main__":
    main()
