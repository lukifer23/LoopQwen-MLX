"""Rescore archived room-number predictions on mutually exclusive path topologies."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from research_archive import read, require, trajectory, verify_terminal


def score(root):
    verify_terminal(root, read(root / "decision.json"))
    records = [read(root / arm / "eval.json.gz")["records"] for arm in ("K1", "K3")]
    groups = {}
    for first, third in zip(*records, strict=True):
        path = [first["start"], *trajectory(first, 3)]
        if path[-1] == path[-2]:
            topology = "stationary_final_step"
        elif len(set(path)) < len(path):
            topology = "other_repeated_state"
        else:
            topology = "four_distinct_states"
        counts = groups.setdefault(
            topology, dict(maps=0, k1_correct=0, k3_correct=0, k3_wins=0, k3_losses=0)
        )
        a, b = [
            r["complete"] and r["visible"].strip() == str(path[-1])
            for r in (first, third)
        ]
        counts["maps"] += 1
        counts["k1_correct"] += a
        counts["k3_correct"] += b
        counts["k3_wins"] += b and not a
        counts["k3_losses"] += a and not b

    require(sum(v["maps"] for v in groups.values()) == len(records[0]), "Cohort")
    return dict(study=root.name, exploratory=True, by_topology=groups)


def shortcuts(root):
    """Compare completed responses with one-edge and start-independent predictions."""
    verify_terminal(root, read(root / "decision.json"))
    summaries = {}
    for arm in ("K1", "K3"):
        records = read(root / arm / "eval.json.gz")["records"]
        counts = Counter()
        expected = Fraction()
        for r in records:
            first, _, target = trajectory(r, 3)
            frequency = Counter(r["map"])
            maximum = max(frequency.values())
            popular = {str(k) for k, v in frequency.items() if v == maximum}
            answer = r["visible"].strip() if r["complete"] else None
            correct = answer == str(target)
            target_popular = str(target) in popular
            counts["maps"] += 1
            counts["one_edge_correct"] += first == target
            counts["model_wins_over_one_edge"] += correct and first != target
            counts["model_losses_to_one_edge"] += not correct and first == target
            counts["model_chose_popular"] += answer in popular
            counts["target_outside_popular"] += not target_popular
            counts["correct_outside_popular"] += correct and not target_popular
            expected += Fraction(int(target_popular), len(popular))
        summaries[arm] = {**counts, "popular_expected_correct": str(expected)}
    return dict(study=root.name, exploratory=True, arms=summaries)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--shortcuts", action="store_true")
    study = Path("research/results/functional-terminal-01")
    report = shortcuts(study) if cli.parse_args().shortcuts else score(study)
    print(json.dumps(report, indent=2))
