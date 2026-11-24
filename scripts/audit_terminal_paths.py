"""Rescore archived room-number predictions on mutually exclusive path topologies."""

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


if __name__ == "__main__":
    print(json.dumps(score(Path("research/results/functional-terminal-01")), indent=2))
