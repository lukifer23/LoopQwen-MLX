"""Post-hoc arithmetic audit of completed first turns, including chained equalities."""

import argparse
import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def amount(expr):
    if len(expr) > 160:
        raise ValueError("Long expression")

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Unsupported expression")

    return walk(ast.parse(expr.strip().replace(",", ""), mode="eval").body)


def check(lines, answer):
    try:
        end = None
        for line in lines:
            terms = line.split("=")
            if len(terms) < 2:
                return "unsupported"
            values = [amount(t) for t in terms]
            if any(v != values[0] for v in values[1:]):
                return "false equality"
            end = values[-1]
        return (
            "consistent"
            if end is not None and end == Fraction(answer)
            else "final mismatch"
        )
    except (ValueError, ArithmeticError, SyntaxError, RecursionError):
        return "unsupported"


def audit():
    root = Path("research/results/gsm-equations-01")
    data = json.loads(Path("research/data/gsm-equations-01.json").read_text())
    for item in data["train"]:
        if check(item["equations"], item["gold"]) != "consistent":
            raise ValueError("Training certificate")
    report = {}
    for arm in ("base", "candidate"):
        by = Counter()
        agreeing = []
        rows = json.loads(gzip.decompress((root / (arm + ".json.gz")).read_bytes()))[
            "records"
        ]
        for r, item in zip(rows, data["dev"], strict=True):
            if r["id"] != item["id"] or r["gold"] != item["gold"]:
                raise ValueError("Math cohort mismatch")
            if r["answer"] is None:
                by["no scorable answer"] += 1
                continue
            equations = r["text"].split("</think>", 1)[0].strip().splitlines()
            result = check(equations, r["answer"])
            by[result] += 1
            if result == "consistent" and r["answer"] == item["gold"]:
                agreeing.append(item["id"])
        report[arm] = dict(
            outcomes=dict(by), arithmetically_consistent_and_correct=agreeing
        )
    return dict(
        scope="Supplementary post-hoc check allowing multiple equal signs. No change to the registered metric. Arithmetic consistency alone does not verify semantic grounding.",
        checked_training_questions=len(data["train"]),
        checksums_sha256=hashlib.sha256(
            (root / "checksums.json").read_bytes()
        ).hexdigest(),
        results=report,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and json.loads(args.output.read_text()) != data:
            raise ValueError("Existing audit differs")
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["results"], indent=2))
