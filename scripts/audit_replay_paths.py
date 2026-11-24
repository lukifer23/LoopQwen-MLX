"""Compare six actual exits with repeating the same model's archived first prediction."""

import json
from pathlib import Path

from research_archive import checksums, read, require, trajectory, verify_replay


def compare(records, cohort):
    scores = {}
    for row, item in zip(records, cohort, strict=True):
        require(row["map_id"] == item["map_id"], "Unpaired map")
        gold = trajectory(item, 6)
        exits = [p.index(max(p)) for p in row["probabilities"]]
        require(len(exits) == len(gold), "Incomplete recurrent trajectory")
        repeated = [exits[0]] * len(gold)
        for name in ("all", "stationary" if gold[-1] == gold[-2] else "moving"):
            counts = scores.setdefault(
                name,
                dict(
                    maps=0,
                    sixth_correct=0,
                    repeated_first_terminal_correct=0,
                    full_correct=0,
                    repeated_first_full_correct=0,
                    full_wins=0,
                    full_losses=0,
                ),
            )
            counts["maps"] += 1
            counts["sixth_correct"] += exits[-1] == gold[-1]
            counts["repeated_first_terminal_correct"] += repeated[-1] == gold[-1]
            counts["full_correct"] += exits == gold
            counts["repeated_first_full_correct"] += repeated == gold
            counts["full_wins"] += exits == gold and repeated != gold
            counts["full_losses"] += exits != gold and repeated == gold
    return scores


if __name__ == "__main__":
    root = Path("research/results/functional-replay-cap84")
    require(checksums(root) == read(root / "checksums.json"), "Archive changed")
    decision = read(root / "decision.json")
    verify_replay(root, decision)
    reg = read(root / "registration.json")
    dataset = Path("research/data") / (reg["dataset"] + ".json.gz")
    cohort = read(dataset)["splits"]["dev"]
    paths = {"parent": root / "initial.json.gz"}
    paths.update(
        {arm: root / "round1" / arm / "eval.json.gz" for arm in ("static", "online")}
    )
    results = {
        arm: compare(read(path)["records"], cohort) for arm, path in paths.items()
    }
    print(json.dumps(results, indent=2))
