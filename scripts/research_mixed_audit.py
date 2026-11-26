"""Blindly credit noncanonical completed numeric or relation answers in mixed deployments."""

import argparse
import gzip
import json
from pathlib import Path

from loopqwen.evaluation.answers import numeric
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read


def audit(study, annotations=None):
    root = Path("research/results") / study
    panel = read(Path("research/data") / (study + ".json"))
    blind = {}
    flags = {}
    for arm in ("candidate", "base"):
        with gzip.open(root / (arm + ".json.gz")) as f:
            rows = json.load(f)["records"]
        if [r["id"] for r in rows] != [r["id"] for r in panel]:
            raise ValueError("Complete paired requests required")
        flags[arm] = []
        for r, item in zip(rows, panel, strict=True):
            valid = r["correct"]
            family = item["family"]
            if r["complete"] and family != "reading":
                visible = (
                    r["text"]
                    .split("</think>", 1)[1]
                    .removesuffix("<|im_end|>")
                    .removesuffix("<|endoftext|>")
                    .strip()
                )
                if family == "entailment":
                    answer = visible.strip("*.! \n\t").lower()
                    canonical = answer in ("yes", "no", "maybe")
                else:
                    answer = numeric(visible)
                    canonical = answer is not None
                if not canonical:
                    key = fingerprint([item["id"], visible])
                    blind[key] = {"family": family, "response": visible}
                    answer = None if annotations is None else annotations.get(key)
                    if answer is not None and not (
                        isinstance(answer, str)
                        and (
                            answer in ("yes", "no", "maybe")
                            if family == "entailment"
                            else numeric(answer) == answer
                        )
                    ):
                        raise ValueError("Invalid normalized annotation")
                valid = answer == item["gold"]
            flags[arm].append(bool(valid))
    if annotations is None:
        return dict(sorted(blind.items()))
    if set(blind) != set(annotations):
        raise ValueError("Label every completed noncanonical response")
    scores = {}
    for family in ("all", *sorted({r["family"] for r in panel})):
        idx = [
            i for i, r in enumerate(panel) if family == "all" or r["family"] == family
        ]
        ds = [int(flags["candidate"][i]) - int(flags["base"][i]) for i in idx]
        scores[family] = {
            "n": len(idx),
            "wins": ds.count(1),
            "losses": ds.count(-1),
            "correct": {arm: sum(v[i] for i in idx) for arm, v in flags.items()},
        }
    return {
        "archive": digest(root / "checksums.json"),
        "reviewed": len(blind),
        "scores": scores,
    }


def qualified(study, annotations):
    scores = audit(study, annotations)["scores"]
    return all(v["wins"] >= v["losses"] for v in scores.values()) and all(
        scores[k]["wins"] > scores[k]["losses"] for k in ("all", "entailment")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.study, read(args.annotations) if args.annotations else None),
            indent=2,
        )
    )
