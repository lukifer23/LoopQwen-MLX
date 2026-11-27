"""Solve the publisher's open-world rule programs again with Z3's Datalog engine."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import z3


def solve(world, question):
    # Keep each quoted argument and each source parenthesis. Parse as lists, independently
    # of the training sampler's extraction and forward-chaining implementation.
    def parse(s):
        stack = [[]]
        for token in re.findall(r'"[^"]*"|\(|\)|->', s):
            if token == "(":
                stack.append([])
            elif token == ")":
                done = stack.pop()
                stack[-1].append(done)
            else:
                stack[-1].append(token.strip('"'))
        assert len(stack) == 1 and len(stack[0]) == 1
        return stack[0][0]

    facts = [parse(r["representation"]) for r in world["triples"].values()]
    rules = [parse(r["representation"]) for r in world["rules"].values()]
    query = parse(question["representation"])
    engine = z3.Fixedpoint()
    engine.set(engine="datalog")
    sort = z3.BitVecSort(8)
    yes, no = (
        z3.Function(k, sort, sort, sort, z3.BoolSort())
        for k in ("support", "refutation")
    )
    engine.register_relation(yes, no)
    constants = {}
    var = z3.Const("individual", sort)
    engine.declare_var(var)

    def literal(a):
        sub, verb, obj, sign = a
        assert sign in ("+", "-", "~")
        terms = []
        for name in (sub, verb, obj):
            if name in ("something", "someone"):
                terms.append(var)
            else:
                n = constants.setdefault(name, len(constants))
                assert n < 256
                terms.append(z3.BitVecVal(n, 8))
        return (yes if sign == "+" else no)(*terms)

    for f in facts:
        engine.fact(literal(f))
    for body, arrow, head in rules:
        assert arrow == "->"
        engine.rule(literal(head), [literal(v) for v in body])
    op = [*query[:3], "-" if query[-1] == "+" else "+"]
    lhs, rhs = (engine.query(literal(v)) for v in (query, op))
    assert (
        lhs in (z3.sat, z3.unsat)
        and rhs in (z3.sat, z3.unsat)
        and not (lhs == rhs == z3.sat)
    )
    return (
        ("true" if lhs == z3.sat else "false" if rhs == z3.sat else "uncertain"),
        str(lhs),
        str(rhs),
    )


def run():
    spec = json.loads(
        Path("research/experiments/proof-curriculum-01.json").read_bytes()
    )
    corpus = Path(spec["training"])
    material = json.loads(corpus.read_bytes())
    original = Path("data/research_external/proofwriter-V2020.12.3.zip")
    with original.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == spec["sources"]["sha256"]
    selected = {r["theory"]: r for r in material["proof"]}
    trace = []
    with ZipFile(original) as z, z.open(spec["sources"]["member"]) as stream:
        for line in stream:
            w = json.loads(line)
            if w["id"] not in selected:
                continue
            row = selected[w["id"]]
            q = w["questions"][row["question"]]
            label, support, opposite = solve(w, q)
            assert row["label"] == label and row["claim"] == q["question"]
            assert row["evidence"].splitlines() == [
                f"{i}. {v['text']}"
                for i, v in enumerate([*w["triples"].values(), *w["rules"].values()], 1)
            ]
            trace.append(
                dict(
                    id=row["id"],
                    oracle=label,
                    query=support,
                    opposite=opposite,
                    world_sha256=hashlib.sha256(line).hexdigest(),
                )
            )
    assert len(trace) == len(selected) == 192
    report = dict(
        version=z3.get_version_string(),
        source=material["source_archive"],
        training=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        worlds=len(trace),
        labels=Counter(r["oracle"] for r in trace),
        results=trace,
    )
    out = Path("docs/measurements/proof-curriculum-source-oracle.json")
    if out.exists():
        assert json.loads(out.read_bytes()) == report
    else:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["version"], report["worlds"], report["labels"])


if __name__ == "__main__":
    run()
