"""Conservative verification of V2/V3 generated procedural certificates."""

import ast
import operator
import re
from fractions import Fraction

from .answers import MARKER, numeric

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


def expression(text):
    """Bounded exact evaluation of the registered integer/rational grammar."""
    if len(text) > 180:
        raise ValueError("long expression")
    tree = ast.parse(text.replace(" mod ", " % ").strip(), mode="eval")

    def visit(n):
        if (
            isinstance(n, ast.Constant)
            and type(n.value) is int
            and n.value.bit_length() < 64
        ):
            return Fraction(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -visit(n.operand) if isinstance(n.op, ast.USub) else visit(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](visit(n.left), visit(n.right))
        raise ValueError("unsupported expression")

    return visit(tree.body)


def obligation(line):
    arr = re.fullmatch(r"(Sorted|Rotated|Selected):\s*\[([-\d,\s]+)\]", line, re.I)
    if arr:
        return (arr[1].lower(), tuple(int(x) for x in arr[2].split(",")), True)
    job = re.fullmatch(r"job (\w+) finishes at (.+)", line, re.I)
    label = ""
    if job:
        label, line = "job " + job[1], job[2]
    else:
        lhs = re.match(r"^(D|x|y)\s*=\s*", line)
        if lhs:
            label, line = lhs[1], line[lhs.end() :]
    parts = line.rsplit("=", 1)
    if len(parts) != 2:
        raise ValueError("unknown step")
    value = expression(parts[0])
    # Only register a bounded scalar literal on a certificate equality.
    rhs = parts[1].strip()
    if len(rhs) > 160 or any(
        abs(int(e)) > 512 for e in re.findall(r"[eE]([+-]?\d+)", rhs)
    ):
        raise ValueError("Unsupported scalar in certificate")
    claimed = Fraction(rhs)
    return (label, re.sub(r"\s+", "", parts[0]), value == claimed)


def graph_obligations(graph, lines):
    from ..data.graphs import solve

    duration = graph["durations"]
    parents = graph["predecessors"]
    target = graph["target"]
    finishes, _, _ = solve(duration, parents, target)
    required = set()

    def include(node):
        if node not in required:
            required.add(node)
            for parent in parents[node]:
                include(parent)

    include(target)
    established = set()
    checks = []
    for line in lines:
        line = re.sub(r"^Step \d+\s*:\s*", "", line, flags=re.I)
        m = re.fullmatch(
            r"job (\w+) finishes at (\d{1,12})\s*\+\s*(\d{1,12})\s*=\s*(\d{1,12})\.?",
            line,
            re.I,
        )
        if not m or m[1].upper() not in duration:
            checks.append("unsupported")
            continue
        node = m[1].upper()
        left, right, claimed = map(int, m.groups()[1:])
        if left + right != claimed:
            checks.append("arithmetic_error")
        elif sorted((left, right)) != sorted(
            (finishes[node] - duration[node], duration[node])
        ):
            checks.append("incorrect_state_transition")
        else:
            established.add(node)
            checks.append("verified")
    complete = required.issubset(established) and all(c == "verified" for c in checks)
    return checks, complete, len(required)


def verify(row, text, truncated=False):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    finals = [(i, numeric(s)) for i, s in enumerate(lines) if re.match(MARKER, s, re.I)]
    gold = numeric(str(row["answer"]))
    if len({v for _, v in finals if v is not None}) > 1:
        final = "conflicting_final_answers"
    elif (
        not finals
        or finals[-1][0] != len(lines) - 1
        or any(v is None for _, v in finals)
    ):
        final = "missing_or_malformed_final"
    else:
        final = "correct" if finals[-1][1] == gold else "incorrect"
    actual = [s for s in lines if not re.match(MARKER, s, re.I)]
    if "transition" in row:
        spec = row["transition"]
        node = spec["start"]
        checks = []
        for line in actual:
            match = re.fullmatch(r"Next\((\d+)\)\s*=\s*(\d+)", line, re.I)
            expected = spec["mapping"][node]
            checks.append(
                "unsupported"
                if not match
                else "verified"
                if tuple(map(int, match.groups())) == (node, expected)
                else "incorrect_state_transition"
            )
            node = expected
        expected_count = spec["n"]
        complete = len(checks) == expected_count and all(
            s == "verified" for s in checks
        )
    else:
        expected = [obligation(s) for s in row["trace"].splitlines()[:-1]]
        checks = []
        for i, line in enumerate(actual):
            try:
                item = obligation(re.sub(r"^Step \d+\s*:\s*", "", line, flags=re.I))
            except (ValueError, SyntaxError, ZeroDivisionError):
                checks.append("unsupported")
                continue
            if not item[2]:
                checks.append("arithmetic_error")
            elif i < len(expected) and item[:2] == expected[i][:2]:
                checks.append("verified")
            elif i < len(expected) and item[0] and item[0] == expected[i][0]:
                checks.append("incorrect_state_transition")
            else:
                checks.append("unsupported")
        complete = len(checks) == len(expected) and all(c == "verified" for c in checks)
        expected_count = len(expected)
    if "graph" in row:
        checks, complete, expected_count = graph_obligations(row["graph"], actual)
    return dict(
        final=final,
        verified=complete and final == "correct",
        all_steps_verified=complete,
        format_compliant=len(finals) == 1 and final in ("correct", "incorrect"),
        truncated=truncated,
        recognized=sum(s != "unsupported" for s in checks),
        correct_steps=checks.count("verified"),
        expected_steps=expected_count,
        checks=checks,
    )
