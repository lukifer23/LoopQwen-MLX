"""An exact, bounded arithmetic tool with references to earlier results."""

import ast
import re
from fractions import Fraction
from operator import add, mul, sub, truediv

from .tool_session import Tool

OPS = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv}


def execute(steps):
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        raise ValueError("Expected 1 through 12 calculations")
    results = []

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(str(node.value))
        elif isinstance(node, ast.Name) and re.fullmatch(r"r[1-9][0-9]?", node.id):
            i = int(node.id[1:]) - 1
            if i >= len(results):
                raise ValueError("Unknown calculation result")
            value = results[i]
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            value = OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        else:
            raise ValueError("Use numbers, arithmetic, parentheses or a prior result")
        if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
            raise ValueError("Calculation exceeds the supported magnitude")
        return value

    for step in steps:
        if (
            not isinstance(step, str)
            or len(step) > 160
            or not re.fullmatch(r"[0-9r.+*/()\s-]+", step)
        ):
            raise ValueError("Invalid calculation text")
        results.append(evaluate(ast.parse(step.strip(), mode="eval").body))
    return results


def calculate(steps):
    values = execute(steps)
    return {"results": [str(x) for x in values], "value": str(values[-1])}


def tool():
    schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 160},
                "description": "Arithmetic steps. In a later step r1 is the first result, r2 the second, and so on.",
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }
    return Tool(
        "calculate",
        "Evaluate a list of arithmetic expressions using numbers, +, -, *, / and parentheses. Returns exact intermediate values and final value.",
        schema,
        calculate,
    )
