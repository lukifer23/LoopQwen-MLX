"""Review completed, unscored arithmetic answers without exposing model identity or gold."""

import argparse
import gzip
import json
from pathlib import Path

from loopqwen.evaluation.answers import numeric
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.evaluation.statistics import exact_mcnemar


def material(study):
    root = Path("research/results") / study
    panel = read(Path("research/data") / (study + ".json"))
    prompts = {r["id"]: r["question"] for r in panel}
    blind, records = {}, {}
    for arm in ("base", "candidate"):
        with gzip.open(root / (arm + ".json.gz")) as fh:
            rows = json.load(fh)["records"]
        if [r["id"] for r in rows] != [r["id"] for r in panel]:
            raise ValueError("Missing or unpaired first turns")
        records[arm] = rows
        for r in rows:
            if r["complete"] and r["answer"] is None:
                key = fingerprint([r["id"], r["visible"]])
                blind[key] = {"question": prompts[r["id"]], "response": r["visible"]}
    return records, dict(sorted(blind.items())), digest(root / "checksums.json")


def score(study, annotations):
    records, blind, archive = material(study)
    if set(annotations) != set(blind) or any(
        value is not None and (not isinstance(value, str) or numeric(value) != value)
        for value in annotations.values()
    ):
        raise ValueError(
            "Label every completed rejected response with one normalized number or null"
        )
    flags = {}
    for arm, rows in records.items():
        flags[arm] = []
        for row in rows:
            answer = row["answer"]
            if row["complete"] and answer is None:
                answer = annotations[fingerprint([row["id"], row["visible"]])]
            flags[arm].append(row["complete"] and answer == row["gold"])
    diff = [
        int(c) - int(b) for c, b in zip(flags["candidate"], flags["base"], strict=True)
    ]
    wins, losses = diff.count(1), diff.count(-1)
    return {
        "study": study,
        "archive": archive,
        "reviewed": len(blind),
        "n": len(diff),
        "correct": {k: sum(v) for k, v in flags.items()},
        "wins": wins,
        "losses": losses,
        "paired_p": exact_mcnemar(wins, losses),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.annotations:
        result = score(args.study, read(args.annotations))
    else:
        _, result, _ = material(args.study)
    print(json.dumps(result, indent=2))
