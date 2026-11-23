"""Paired seed and stratified-item resampling uncertainty for the registered final study."""

import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import read, save


def exact_mcnemar(a, b):
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def paired(left, right):
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("IDs differ")
    diff = np.array(
        [int(r["correct"]) - int(l["correct"]) for l, r in zip(left, right)]
    )
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    return diff, dict(
        dense=sum(r["correct"] for r in left),
        loop=sum(r["correct"] for r in right),
        wins=wins,
        losses=losses,
        n=len(left),
        difference=float(diff.mean()),
        mcnemar=exact_mcnemar(wins, losses),
    )


def summarize(name="phase2_confirm"):
    m = read(Path("experiments") / (name + ".json"))
    root = Path("outputs/phase2") / name
    result = {}
    for split in m["splits"]:
        for protocol in ("answer", "trace"):
            for control in ("dense", "time"):
                items = []
                ids = []
                per_seed = []
                groups = defaultdict(list)
                strata = defaultdict(list)
                for left, right in m["matched"]:
                    if control == "time":
                        left = left.removesuffix("_dense") + "_time"
                    a, b = [read(root / (run + ".json")) for run in (left, right)]
                    aa, bb = [
                        entry["scores"][split][protocol]["generations"]["records"]
                        for entry in (a, b)
                    ]
                    diff, s = paired(aa, bb)
                    if items and ids != [r["id"] for r in aa]:
                        raise ValueError("Seed pair uses different items")
                    if not items:
                        ids = [r["id"] for r in aa]
                        for i, r in enumerate(aa):
                            strata[r["category"]].append(i)
                    for i, r in enumerate(aa):
                        for key in ("category", "complexity", "steps"):
                            groups[f"{key}={r[key]}"].append(
                                [int(r["correct"]), int(bb[i]["correct"])]
                            )
                    items.append(diff)
                    per_seed.append(s | {"seed": a["training"]["seed"]})
                arr = np.stack(items)
                rng = np.random.default_rng(773)
                dist = []
                for _ in range(4000):
                    seed_idx = rng.integers(len(items), size=len(items))
                    indices = np.concatenate(
                        [
                            rng.choice(idx, len(idx), replace=True)
                            for idx in strata.values()
                        ]
                    )
                    dist.append(arr[seed_idx][:, indices].mean())
                summary = {}
                for arm in ("dense", "loop"):
                    acc = [s[arm] / s["n"] for s in per_seed]
                    summary[arm] = dict(
                        mean=statistics.mean(acc), sd=statistics.stdev(acc)
                    )
                result[f"{split}/{protocol}/{control}"] = dict(
                    seeds=per_seed,
                    accuracy=summary,
                    mean_difference=float(arr.mean()),
                    paired_ci95=np.percentile(dist, [2.5, 97.5]).tolist(),
                    groups={
                        k: dict(
                            items_per_seed=len(v) // len(items),
                            dense_accuracy=float(np.mean(v, axis=0)[0]),
                            loop_accuracy=float(np.mean(v, axis=0)[1]),
                        )
                        for k, v in groups.items()
                    },
                )
    save(root / "paired.json", result)
    return result


if __name__ == "__main__":
    summarize()
