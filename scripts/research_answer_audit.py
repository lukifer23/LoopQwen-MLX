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


def reviewed(study, annotations):
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
    return records, flags, blind, archive


def score(study, annotations):
    _, flags, blind, archive = reviewed(study, annotations)
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


def combined(first, confirmation, initial_annotations, final_annotations):
    original = Path("research/results") / first
    repeat = Path("research/results") / confirmation
    spec = read(repeat / "registration.json")
    prior = read(original / "registration.json")
    if (
        spec["qualification"] != digest(original / "checksums.json")
        or spec["model"] != str(original / "model")
        or spec["weights"] != read(original / "decision.json")["weights"]
        or any(
            spec[k] != prior[k] for k in ("switch_after", "generation_cap", "sources")
        )
    ):
        raise ValueError(
            "Pooling requires a preregistered unchanged-checkpoint replication"
        )
    ids = set()
    grouped = {}
    for study, annotations in (
        (first, initial_annotations),
        (confirmation, final_annotations),
    ):
        records, flags, _, _ = reviewed(study, annotations)
        panel = read(Path("research/data") / (study + ".json"))
        for i, item in enumerate(panel):
            if item["id"] in ids:
                raise ValueError("Repeated pooled question")
            ids.add(item["id"])
            for name in ("all", item["source"]):
                group = grouped.setdefault(
                    name,
                    {
                        "strict": [],
                        "content": [],
                        "seconds": {"candidate": 0, "base": 0},
                    },
                )
                for arm in group["seconds"]:
                    group["seconds"][arm] += records[arm][i]["seconds"]
                group["strict"].append(
                    [bool(records[a][i]["correct"]) for a in ("candidate", "base")]
                )
                group["content"].append(
                    [bool(flags[a][i]) for a in ("candidate", "base")]
                )
    for group in grouped.values():
        group["n"] = len(group["strict"])
        for key in ("strict", "content"):
            pairs = group[key]
            wins = pairs.count([True, False])
            losses = pairs.count([False, True])
            group[key] = dict(
                candidate=sum(c for c, _ in pairs),
                base=sum(b for _, b in pairs),
                wins=wins,
                losses=losses,
                paired_p=exact_mcnemar(wins, losses),
            )
    return {"studies": [first, confirmation], "scores": grouped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--replication")
    parser.add_argument("--replication-annotations", type=Path)
    args = parser.parse_args()
    if args.replication or args.replication_annotations:
        if not (args.replication and args.annotations and args.replication_annotations):
            parser.error("Pooling needs both studies and both annotations")
        result = combined(
            args.study,
            args.replication,
            read(args.annotations),
            read(args.replication_annotations),
        )
    elif args.annotations:
        result = score(args.study, read(args.annotations))
    else:
        _, result, _ = material(args.study)
    print(json.dumps(result, indent=2))
