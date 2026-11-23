"""Aggregate frozen generation records by task family, shift and solver properties."""

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import read, save


def difficulty(row):
    family = row["category"]
    lines = row["trace"].splitlines()[:-1]
    operations = 0
    for line in lines:
        if "=" not in line:
            continue
        expr = line.rsplit("=", 1)[0].split("=")[-1].strip().replace(" mod ", " % ")
        expr = re.sub(r"^job \w+ finishes at ", "", expr)
        operations += sum(
            isinstance(n, ast.BinOp) for n in ast.walk(ast.parse(expr, mode="eval"))
        )
    if "graph" in row:
        depth = row["difficulty"]["dependency_depth"]
        variables = row["difficulty"]["nodes"]
    elif family == "system":
        depth = 4  # Registered D -> x -> y -> requested scalar certificate.
        variables = 2
    elif family == "array":
        depth = 4  # Sort, rotate, strided selection, reduction.
        variables = 1
    else:
        depth = len(lines)
        variables = len(lines) if family == "dependencies" else 1
    return dict(
        certificate_arithmetic_ops=operations,
        transition_depth=depth,
        variables=variables,
        intermediate_states=len(lines),
        graph=row.get("difficulty"),
    )


def run():
    root = Path("outputs/phase3")
    results = read(root / "fresh-development.json")
    features = {}
    index = {}
    for split, report in results.items():
        docs = {
            r["id"]: r
            for r in map(
                json.loads, Path("data/v3", split + ".jsonl").read_text().splitlines()
            )
        }
        for row in docs.values():
            features[row["id"]] = difficulty(row)
        for label, measurement in report["metrics"].items():
            arm, mode = label.split("/")
            path = (
                root / "eval/100000" / arm / mode / (measurement["eval_id"] + ".json")
            )
            rec = read(path)
            groups = defaultdict(Counter)
            for item in rec["generations"]["records"]:
                row = docs[item["id"]]
                d = features[item["id"]]
                c = item["verification"]
                for key in (
                    "family/" + row["category"],
                    "shift/" + row["shift"],
                    "complexity/" + str(row["complexity"]),
                    "depth/" + str(d["transition_depth"]),
                    "family-depth/"
                    + row["category"]
                    + "/"
                    + str(d["transition_depth"]),
                ):
                    group = groups[key]
                    group.update(
                        n=1,
                        correct=int(item["correct"]),
                        verified=int(c["verified"]),
                        truncated=int(c["truncated"]),
                        recognized=c["recognized"],
                        correct_steps=c["correct_steps"],
                    )
                    group.update("step_" + status for status in c["checks"])
            index[split + "/" + label] = {key: dict(v) for key, v in groups.items()}
    save(root / "difficulty.json", dict(features=features, measurement_groups=index))


if __name__ == "__main__":
    run()
