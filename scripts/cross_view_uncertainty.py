"""Preserve within-article dependence when describing cross-view reading differences."""

from collections import defaultdict

import numpy as np

from loopqwen.evaluation.extractive import normalized
from loopqwen.evaluation.statistics import exact_mcnemar


def paired_articles(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["family"] != "retention":
            groups[r["cluster"]].append(r)
    assert len(groups) == 20
    metrics = defaultdict(list)
    samples = defaultdict(list)
    for article, items in groups.items():
        assert len(items) == 4
        by_view = {
            f: [r for r in items if r["family"] == f] for f in ("file", "inline")
        }
        assert all(len(pair) == 2 for pair in by_view.values())
        for arm in ("candidate", "control"):
            get = lambda r, key: r[arm + "_" + key]
            metrics[arm, "all_four"].append(all(get(r, "success") for r in items))
            for family, pair in by_view.items():
                metrics[arm, family].append(sum(get(r, "success") for r in pair) / 2)
                metrics[arm, family + "_f1"].append(sum(get(r, "f1") for r in pair) / 2)
                if all(get(r, "completed") for r in pair):
                    repeated = len({normalized(get(r, "answer")) for r in pair}) == 1
                    samples[arm, family].append(
                        {"article": article, "repeated_answer": repeated}
                    )
    rng = np.random.default_rng(62889)
    indices = rng.integers(len(groups), size=(20000, len(groups)))
    output = {}
    for key in ("all_four", "file", "inline", "file_f1", "inline_f1"):
        left, right = (
            np.asarray(metrics[arm, key], dtype=float)
            for arm in ("candidate", "control")
        )
        ds = left - right
        interval = np.percentile(ds[indices].mean(axis=1), [2.5, 97.5])
        output[key] = {"difference": float(ds.mean()), "interval": interval.tolist()}
        if key == "all_four":
            wins, losses = int((ds > 0).sum()), int((ds < 0).sum())
            output[key].update(
                wins=wins, losses=losses, paired_p=exact_mcnemar(wins, losses)
            )
    return {
        "articles": len(groups),
        "draws": len(indices),
        "seed": 62889,
        "paired": output,
        "two_completed": {
            arm: {f: samples[arm, f] for f in ("file", "inline")}
            for arm in ("candidate", "control")
        },
    }
