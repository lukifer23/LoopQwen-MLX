"""An exact, bounded arithmetic tool with references to earlier results."""

import ast
import re
from fractions import Fraction
from operator import add, mul, sub, truediv

from .tool_session import Tool

OPS = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv}


def execute(steps, *, check_equalities=False):
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        raise ValueError("Expected 1 through 12 calculations")
    results = []

    def evaluate(node, expression):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(ast.get_source_segment(expression, node))
        elif isinstance(node, ast.Name) and re.fullmatch(r"r[1-9][0-9]?", node.id):
            i = int(node.id[1:]) - 1
            if i >= len(results):
                raise ValueError("Unknown calculation result")
            value = results[i]
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = evaluate(node.operand, expression)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            value = OPS[type(node.op)](
                evaluate(node.left, expression), evaluate(node.right, expression)
            )
        else:
            raise ValueError("Use numbers, arithmetic, parentheses or a prior result")
        if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
            raise ValueError("Calculation exceeds the supported magnitude")
        return value

    def expression_value(expression):
        expression = expression.strip()
        if not re.fullmatch(r"[0-9r.+*/()\s-]+", expression):
            raise ValueError("Invalid calculation text")
        return evaluate(ast.parse(expression, mode="eval").body, expression)

    for step in steps:
        if not isinstance(step, str) or len(step) > 160:
            raise ValueError("Invalid calculation text")
        parts = step.split("=") if check_equalities else [step]
        value = expression_value(parts[0])
        for claim in parts[1:]:
            if expression_value(claim) != value:
                raise ValueError("Arithmetic equality is false")
        results.append(value)
    return results


def calculate(steps):
    values = execute(steps)
    return {"results": [str(x) for x in values], "value": str(values[-1])}


def checked_calculate(steps):
    values = execute(steps, check_equalities=True)
    return {"results": [str(x) for x in values], "value": str(values[-1])}


def tool(*, check_equalities=False):
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
    description = "Evaluate a list of arithmetic expressions using numbers, +, -, *, / and parentheses. Returns exact intermediate values and final value."
    if check_equalities:
        description += " A step may include an equality. Every side must evaluate to the same exact number."
    return Tool(
        "calculate",
        description,
        schema,
        checked_calculate if check_equalities else calculate,
    )
