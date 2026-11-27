"""Recompute every sampled example, actual learner-token contrast and native training checkpoint."""

import argparse
import random
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.linked_rollin_data import encode


def run(root):
    spec = read(root / "registration.json")
    fit = read(root / "fit.json")
    views = unpack(root / "training-views.json.gz")
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    by = {}
    for w in views:
        prefix, labels, _, wrong = encode(tok, w)
        count = (
            sum(bool(tok.decode([x]).strip()) for x in labels[: labels.index(close)])
            if close in labels
            else 0
        )
        if wrong:
            j, b = wrong
            actual = w["learner"]
            cut = actual.index(close) + 1
            assert (
                prefix[-cut:] == actual[:cut]
                and labels[:j] == actual[cut : cut + j]
                and labels[j] != b == actual[cut + j]
            )
        by[w["id"]] = len(labels), count, int(wrong is not None)
    assert len(by) == len(views)
    totals = [0, 0, 0]
    draws = Counter()
    seen = set()
    points = []
    for phase in range(2):
        rng = random.Random(spec["fit_seed"] + phase)
        groups = [
            [
                w["id"]
                for w in views
                if w["kind"] == k and (phase or "-round-1-turn-" not in w["id"])
            ]
            for k in ("correction", "action", "answer", "retained")
        ]
        for step in range(phase * 400 + 1, (phase + 1) * 400 + 1):
            for k, group in zip(
                ("correction", "action", "answer", "retained"), groups, strict=True
            ):
                uid = rng.choice(group)
                seen.add(uid)
                draws[k] += 1
                totals = [a + b for a, b in zip(totals, by[uid], strict=True)]
            if step % 100 == 0:
                points.append(
                    dict(
                        zip(
                            (
                                "step",
                                "targets",
                                "explanation_targets",
                                "contrast_draws",
                            ),
                            (step, *totals),
                            strict=True,
                        )
                    )
                )
    assert totals == [
        fit[k] for k in ("targets", "explanation_targets", "contrast_draws")
    ]
    assert dict(draws) == fit["draws"] and points == [
        {k: v[k] for k in points[0]} for v in read(root / "optimizer-steps.json")
    ]
    out = {
        "training_views": digest(root / "training-views.json.gz"),
        "trajectories": [digest(root / f"student-round-{i}.json.gz") for i in range(2)],
        "unique_visited_corrections": sum(v["kind"] == "correction" for v in views),
        "distinct_positive_views_drawn": len(seen),
        "draws": dict(draws),
        "checkpoints": points,
    }
    path = Path("docs/measurements/linked-rollin-exposure.json")
    if path.exists():
        assert read(path) == out
    else:
        save(path, out)
    print(
        "Verified teacher boundaries and actual competing emitted tokens on",
        sum(draws.values()),
        "draws.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("research/results/linked-rollin-01")
    )
    run(p.parse_args().root)
