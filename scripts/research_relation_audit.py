"""Review completed noncanonical entailment judgments without seeing arm or gold."""

import argparse
import gzip
import json
from pathlib import Path

from loopqwen.evaluation.relations import premise_randomization
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read


def audit(study, annotations=None, *, paired=False):
    path = Path("research/results") / study
    panel = read(Path("research/data") / (study + ".json"))
    prompts = {r["id"]: r for r in panel}
    blind = {}
    by = {}
    for arm in ("candidate", "base"):
        with gzip.open(path / (arm + ".json.gz")) as stream:
            records = json.load(stream)["records"]
        if [r["id"] for r in records] != [r["id"] for r in panel]:
            raise ValueError("The complete paired panel is required")
        flags = []
        for r in records:
            item = prompts[r["id"]]
            label = None
            if r["complete"]:
                canonical = r["visible"].strip().strip("*.! \n\t").lower()
                if canonical in ("yes", "no", "maybe"):
                    label = canonical
                else:
                    key = fingerprint([r["id"], r["visible"]])
                    blind[key] = {
                        "premise": item["premise"],
                        "claim": item["hypothesis"],
                        "response": r["visible"],
                    }
                    if annotations is not None:
                        label = annotations.get(key)
            flags.append(label == item["label"])
        by[arm] = flags
    if annotations is None:
        return dict(sorted(blind.items()))
    if set(annotations) != set(blind) or any(
        v not in ("yes", "no", "maybe", None) for v in annotations.values()
    ):
        raise ValueError(
            "Every noncanonical completed answer needs a blind human-readable judgment"
        )
    if paired:
        return panel, by
    return {
        "study": study,
        "archive": digest(path / "checksums.json"),
        "reviewed": len(blind),
        "scores": summarize(panel, by),
    }


def summarize(panel, by):
    result = {}
    for source in ("all", "esnli", "multinli"):
        idx = [
            i for i, r in enumerate(panel) if source == "all" or r["source"] == source
        ]
        diffs = [int(by["candidate"][i]) - int(by["base"][i]) for i in idx]
        result[source] = {
            "n": len(idx),
            "correct": {k: sum(v[i] for i in idx) for k, v in by.items()},
            "wins": diffs.count(1),
            "losses": diffs.count(-1),
            "premise_p": premise_randomization(diffs),
        }
    return result


def combined(first, repeat, original_annotations, repeat_annotations):
    base_dir = Path("research/results") / first
    repeat_dir = Path("research/results") / repeat
    reg = read(repeat_dir / "registration.json")
    if (
        reg["qualification"] != digest(base_dir / "checksums.json")
        or reg["weights"] != read(base_dir / "decision.json")["weights"]
        or reg["weights"] != read(repeat_dir / "decision.json")["weights"]
    ):
        raise ValueError("Unchanged, preregistered confirmation required")
    p, a = audit(first, original_annotations, paired=True)
    q, b = audit(repeat, repeat_annotations, paired=True)
    if {r["premise"].casefold() for r in p}.intersection(
        r["premise"].casefold() for r in q
    ):
        raise ValueError("Pooled premise clusters overlap")
    return {
        "studies": [first, repeat],
        "scores": summarize(p + q, {arm: a[arm] + b[arm] for arm in a}),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study")
    p.add_argument("--annotations", type=Path)
    p.add_argument("--replication")
    p.add_argument("--replication-annotations", type=Path)
    args = p.parse_args()
    if args.replication or args.replication_annotations:
        if not (args.replication and args.annotations and args.replication_annotations):
            p.error("Pooling requires both studies and both annotation files")
        result = combined(
            args.study,
            args.replication,
            read(args.annotations),
            read(args.replication_annotations),
        )
    else:
        result = audit(args.study, read(args.annotations) if args.annotations else None)
    print(json.dumps(result, indent=2))
