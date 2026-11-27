"""Independently check the published FOLIO answers with their source logical formulas."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import z3

NAME = "reasoned-evidence-01"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[∀∃¬∧∨⊕→↔(),]|\s+")
PRECEDENCE = {"↔": 1, "→": 2, "⊕": 3, "∨": 4, "∧": 5}
OPS = {"↔": lambda a, b: a == b, "→": z3.Implies, "⊕": z3.Xor, "∨": z3.Or, "∧": z3.And}


class Language:
    def __init__(self):
        self.entity = z3.DeclareSort("Entity")
        self.predicates = {}
        self.constants = {}

    def parse(self, source):
        tokens = TOKEN.findall(source)
        if "".join(tokens) != source:
            raise ValueError("Unsupported source symbol: " + source)
        tokens = [t for t in tokens if t.strip()]
        at = 0

        def take(expected=None):
            nonlocal at
            if at == len(tokens):
                raise ValueError("Unexpected end: " + source)
            item = tokens[at]
            at += 1
            if expected is not None and item != expected:
                raise ValueError((item, expected, source))
            return item

        def expr(level, bound):
            head = take()
            if head in ("∀", "∃"):
                name = take()
                atom = z3.Const(name, self.entity)
                if tokens[at] not in ("(", "∀", "∃"):
                    raise ValueError("Unbounded quantifier scope")
                body = expr(6, {**bound, name: atom})
                left = (z3.ForAll if head == "∀" else z3.Exists)([atom], body)
            elif head == "¬":
                left = z3.Not(expr(6, bound))
            elif head == "(":
                left = expr(1, bound)
                take(")")
            else:
                if not head.isidentifier():
                    raise ValueError("Expected predicate: " + head)
                take("(")
                args = []
                while True:
                    name = take()
                    if not name.isidentifier():
                        raise ValueError("Expected logical term: " + name)
                    if name in bound:
                        args.append(bound[name])
                    else:
                        if name not in self.constants:
                            self.constants[name] = z3.Const(name, self.entity)
                        args.append(self.constants[name])
                    sep = take()
                    if sep == ")":
                        break
                    if sep != ",":
                        raise ValueError("Expected comma: " + source)
                if head not in self.predicates:
                    self.predicates[head] = z3.Function(
                        head, *([self.entity] * len(args)), z3.BoolSort()
                    )
                left = self.predicates[head](*args)
            while at < len(tokens) and PRECEDENCE.get(tokens[at], 0) >= level:
                operator = take()
                p = PRECEDENCE[operator]
                right = expr(p if operator == "→" else p + 1, bound)
                left = OPS[operator](left, right)
            return left

        result = expr(1, {})
        if at != len(tokens):
            raise ValueError("Unconsumed formula: " + source)
        return result


def audit():
    spec = json.loads(Path("research/experiments", NAME + ".json").read_text())
    src = spec["sources"]["folio_validation"]
    original = Path(src["path"]).read_bytes()
    assert hashlib.sha256(original).hexdigest() == src["sha256"]
    by_key = {
        (tuple(v.strip() for v in r["premises"]), r["conclusion"]): r
        for r in map(json.loads, original.splitlines())
    }
    panel = json.loads(Path(spec["material"]).read_text())
    rows = []
    for w in panel["rows"]:
        if w["family"] != "inline":
            continue
        premises, tail = w["questions"][0].split("\n\nClaim: ", 1)
        natural = [re.sub(r"^\d+\. ", "", p) for p in premises.splitlines()]
        r = by_key[tuple(natural), tail.rsplit("\nDoes the evidence", 1)[0]]
        lang = Language()
        assumptions = [lang.parse(p) for p in r["premises-FOL"]]
        claim = lang.parse(r["conclusion-FOL"])
        checks = []
        for condition in (claim, z3.Not(claim)):
            solver = z3.Solver()
            solver.set(timeout=4000)
            solver.add(*assumptions, condition)
            checks.append(str(solver.check()))
        label = {
            ("sat", "unsat"): "true",
            ("unsat", "sat"): "false",
            ("sat", "sat"): "uncertain",
            ("unsat", "unsat"): "inconsistent",
        }.get(tuple(checks), "unresolved")
        rows.append(
            {
                "id": w["id"],
                "cluster": w["cluster"],
                "publisher": w["gold"],
                "formal": label,
                "agrees": label == w["gold"],
                "satisfiable_with_claim": checks[0],
                "satisfiable_with_negation": checks[1],
                "premises_fol": r["premises-FOL"],
                "conclusion_fol": r["conclusion-FOL"],
                "smtlib_premises": [a.sexpr() for a in assumptions],
                "smtlib_conclusion": claim.sexpr(),
            }
        )
    assert len(rows) == 36
    return {
        "version": z3.get_version_string(),
        "source_sha256": src["sha256"],
        "worlds": len({r["cluster"] for r in rows}),
        "answers": len(rows),
        "agreements": sum(r["agrees"] for r in rows),
        "formal_verdicts": dict(Counter(r["formal"] for r in rows)),
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/measurements/reasoned-evidence-source-oracle.json"),
    )
    args = p.parse_args()
    result = audit()
    if args.output.exists():
        assert json.loads(args.output.read_text()) == result
    else:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print({k: v for k, v in result.items() if k != "rows"})
