"""Freeze the first legal equal-width full-history key swap in a declared seed sequence."""

import argparse

from mlx_lm.utils import load_tokenizer

from loopqwen.io import save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.counterfactual_memory import material


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("seed_length", nargs="+", help="seed/length pairs")
    args = p.parse_args()
    tok = load_tokenizer(local_snapshot())
    groups = []
    for target in args.seed_length:
        start, length = map(int, target.split("/"))
        for seed in range(start, start + 30):
            try:
                group = material(tok, dict(seed=seed, minimum_length=length))
            except ValueError as exc:
                if "swap" not in str(exc).lower():
                    raise
                continue
            groups.append(group)
            print(seed, [len(x["prompt_ids"]) for x in group["panel"]], flush=True)
            break
        else:
            raise ValueError("No legal swap in the declared sequence")
    save(args.output, groups)
