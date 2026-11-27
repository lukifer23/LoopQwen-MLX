"""Perturb individual source amounts in real calculator programs and re-execute them exactly."""

import argparse
import ast
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from loopqwen.calculator_tool import OPS, calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.native_function_study import executions
from loopqwen.research.shared_query_features import PRIMARY, RESULTS
from scripts.assistance_deliberation import write_csv
from scripts.shared_query_material import OPERATORS, ROOT, numeric_argument
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/shared-query-calculation-perturbations.json")
SHIFTS = (Fraction(11, 7), Fraction(-5, 13))


def expert(w, chosen=None, shift=0):
    steps = []
    operands = set()

    def argument(raw):
        value = numeric_argument(raw)
        if value.startswith("r"):
            return steps[int(value[1:]) - 1]
        n = Fraction(value)
        if not raw.startswith("const_"):
            operands.add(n)
            if n == chosen:
                n += shift
        return n

    arithmetic = {
        "+": OPS[ast.Add],
        "-": OPS[ast.Sub],
        "*": OPS[ast.Mult],
        "/": OPS[ast.Div],
    }
    for op, a, b in re.findall(r"(\w+)\(([^()]+), ([^()]+)\)", w["program"]):
        steps.append(arithmetic[OPERATORS[op]](argument(a), argument(b)))
    assert len(w["steps"]) in (len(steps), len(steps) + 1)
    if len(w["steps"]) > len(steps):
        assert w["steps"][-1] == f"r{len(steps)}*100"
        steps.append(steps[-1] * 100)
    return steps[-1], operands


def execute_change(steps, chosen, shift):
    # Input has already passed the actual bounded calculator. Keep rN bindings
    # local to this real call. Do not assign hidden causality to pasted numbers.
    results = []

    def evaluate(node, text):
        if isinstance(node, ast.Name):
            return results[int(node.id[1:]) - 1]
        if isinstance(node, ast.Constant):
            value = Fraction(ast.get_source_segment(text, node))
            return value + shift if value == chosen else value
        piece = ast.get_source_segment(text, node)
        # The material represents e.g. 39.2 as the reduced rational 196/5.
        # Recognize that literal as a unit before considering its numerator.
        if re.fullmatch(r"[+-]?\d+(?:\s*/\s*\d+)?", piece):
            compact = "".join(piece.split())
            if compact == str(chosen):
                return chosen + shift
        if isinstance(node, ast.UnaryOp):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return sign * evaluate(node.operand, text)
        return OPS[type(node.op)](evaluate(node.left, text), evaluate(node.right, text))

    for expression in steps:
        text = expression.strip()
        results.append(evaluate(ast.parse(text, mode="eval").body, text))
    return results[-1]


def probe(w, steps):
    target, inputs = expert(w)
    assert target == Fraction(w["observed"]["value"])
    original = Fraction(calculate(steps)["value"])
    checked = []
    for amount in sorted(inputs):
        for shift in SHIFTS:
            wanted, _ = expert(w, amount, shift)
            if wanted == target:
                continue
            actual = None
            try:
                actual = execute_change(steps, amount, shift)
            except ZeroDivisionError:
                pass
            checked.append(
                {
                    "input": str(amount),
                    "change": str(shift),
                    "wanted": str(wanted),
                    "actual": None if actual is None else str(actual),
                    "agreed": actual == wanted,
                }
            )
    return {
        "original_exact": original == target,
        "tests": checked,
        "matches": bool(checked)
        and original == target
        and all(r["agreed"] for r in checked),
    }


def run(report=False):
    source = ROOT / "held.json.gz"
    rows = [w for w in unpack(source) if w["family"] == "report"]
    assert len(rows) == 32
    gold = []
    for w in rows:
        real = probe(w, w["steps"])
        value = Fraction(w["observed"]["value"])
        literal = f"{value.numerator * 1000003}/{value.denominator * 1000003}"
        # A valid dummy sum and a subsequent source-independent rational reproduce the original
        # return. They carry no source-operand dependence for this report.
        sham = probe(w, ["2+3", literal])
        gold.append(
            {
                "id": w["id"],
                "expert_matches": real["matches"],
                "expert_tests": len(real["tests"]),
                "literal_with_unrelated_computation_matches": sham["matches"],
            }
        )
    assert all(
        r["expert_matches"] and not r["literal_with_unrelated_computation_matches"]
        for r in gold
    )
    plan = {
        "held": digest(source),
        "study": digest(Path("research/experiments") / (PRIMARY + ".json")),
        "shifts": [str(x) for x in SHIFTS],
        "question": "Does the model's last successful calculator call still agree with the published program after changing one operative source number at a time? Require exact unrounded financial quantity agreement before and after every perturbation. Test both displacements; omit input changes which leave the expert's final result unchanged. Restrict to self-contained last calls and literal decimals or reduced-rational spellings of the source values. Equivalent unit conversions, rounded results and dependencies across calls can fall outside this strict diagnostic. Every registered score and masked review is reported separately.",
        "expert_controls": gold,
    }
    if not report:
        persist(PLAN, plan)
        return {
            "questions": len(rows),
            "expert_interventions": sum(r["expert_tests"] for r in gold),
        }
    committed(PLAN)
    assert read(PLAN) == plan
    committed(Path("docs/measurements/shared-query-reading.json"))
    result = []
    summary = {}
    archive = RESULTS / PRIMARY
    for arm in ("candidate", "control"):
        c = Counter()
        records = {r["id"]: r for r in unpack(archive / (arm + ".json.gz"))}
        for w in rows:
            (q,) = records[w["id"]]["requests"]
            valid = [
                e
                for e in executions(q["trajectory"])
                if e["call"]["name"] == "calculate" and "value" in e["observation"]
            ]
            analysis = None
            if valid:
                last = valid[-1]
                steps = last["call"]["arguments"]["steps"]
                assert calculate(steps) == last["observation"]
                analysis = probe(w, steps)
            matches = analysis is not None and analysis["matches"]
            c["requests"] += 1
            c["last_call_present"] += bool(valid)
            c["registered_grounded"] += q["score"]["grounded"]
            c["program_matches"] += matches
            c["grounded_and_matches"] += q["score"]["grounded"] and matches
            result.append(
                {
                    "id": w["id"],
                    "arm": arm,
                    "successful_calls": len(valid),
                    "grounded": q["score"]["grounded"],
                    "program_matches": matches,
                    "analysis": analysis,
                }
            )
        summary[arm] = dict(c)
    write_csv(Path("paper/shared-query-calculation-perturbations.csv"), result)
    persist(
        Path("docs/measurements/shared-query-calculation-perturbations.json"),
        {
            "plan": digest(PLAN),
            "archive": digest(archive / "checksums.json"),
            "results": summary,
        },
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    print(run(p.parse_args().report))
