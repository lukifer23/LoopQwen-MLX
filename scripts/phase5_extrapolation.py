"""Compare fixed-depth predictions with start-conditioned training priors by cycle regime."""

from collections import Counter
import json
from pathlib import Path


def predictions(reference, k, wrapped):
    hist = {s: Counter() for s in range(8)}
    for r in reference:
        start = r["start"]
        seen = {start}
        node = start
        repeat = False
        for _ in range(k):
            node = r["map"][node]
            repeat |= node in seen
            seen.add(node)
        if repeat == wrapped:
            hist[start][node] += 1
    return {s: max(range(8), key=lambda j: h[j]) for s, h in hist.items()}


def summarize(result, train):
    report = []
    for k in (4, 5, 6):
        for wrapped in (False, True):
            guess = predictions(train, k, wrapped)
            group = [r for r in result["records"] if (r["cycle"] <= k) == wrapped]
            score = lambda fn: sum(fn(r) for r in group)
            report.append(
                dict(
                    depth=k,
                    wrapped=wrapped,
                    n=len(group),
                    correct=score(lambda r: r["pred"][k - 1] == r["gold"][k - 1]),
                    prior=score(lambda r: guess[r["start"]] == r["gold"][k - 1]),
                    predicts_prior_gold=score(
                        lambda r: r["pred"][k - 1] == r["gold"][k - 2]
                    ),
                    stays_at_own_previous=score(
                        lambda r: r["pred"][k - 1] == r["pred"][k - 2]
                    ),
                    correct_full_path=score(lambda r: r["pred"][:k] == r["gold"][:k]),
                )
            )
    return report


def training_rows():
    return [
        json.loads(s)
        for s in Path("data/phase5_pointer/train.jsonl").read_bytes().splitlines()
    ]
