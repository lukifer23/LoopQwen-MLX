"""Execute and score the numerical prefix of BFCL v3 exec_simple locally."""

import ast
import math
import re
from statistics import fmean, pstdev

from ..tool_session import Tool

INSTRUCTION = (
    " Use the supplied function. Give its returned number only, without units."
)


def probability(n, k, p):
    if not 0 <= k <= n <= 10000 or not 0 <= p <= 1:
        raise ValueError("Invalid binomial parameters")
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def similarity(vectorA, vectorB):
    return math.fsum(x * y for x, y in zip(vectorA, vectorB, strict=True)) / (
        math.hypot(*vectorA) * math.hypot(*vectorB)
    )


def permutations(n, k):
    if not 0 <= k <= n <= 100:
        raise ValueError("Invalid permutation parameters")
    return math.perm(n, k)


FUNCTIONS = {
    "calc_binomial_probability": probability,
    "calculate_cosine_similarity": similarity,
    "calculate_density": lambda mass, volume: mass / volume,
    "calculate_displacement": lambda initial_velocity, acceleration, time: (
        initial_velocity * time + acceleration * time**2 / 2
    ),
    "calculate_electrostatic_potential_energy": lambda charge, voltage: (
        charge * voltage
    ),
    "calculate_final_velocity": lambda initial_velocity, acceleration, time: (
        initial_velocity + acceleration * time
    ),
    "calculate_future_value": lambda present_value, interest_rate, periods: (
        present_value * (1 + interest_rate) ** periods
    ),
    "calculate_mean": lambda numbers: fmean(numbers),
    "calculate_permutations": permutations,
    "calculate_standard_deviation": lambda numbers: pstdev(numbers),
    "calculate_triangle_area": lambda base, height: base * height / 2,
}


def definition(row):
    # Preserve descriptions, requirements and optional defaults. Map the publisher's type vocabulary.
    from copy import deepcopy

    tool = deepcopy(row["function"][0])

    def convert(node):
        node["type"] = {"dict": "object", "float": "number"}.get(
            node["type"], node["type"]
        )
        for child in node.get("properties", {}).values():
            convert(child)
        if "items" in node:
            convert(node["items"])

    convert(tool["parameters"])
    return tool


def tools_for(row):
    f = definition(row)

    def execute(**kwargs):
        result = FUNCTIONS[f["name"]](**kwargs)
        if not math.isfinite(result):
            raise ValueError("Non-finite result")
        return {"value": float(format(result, ".7g"))}

    return [Tool(f["name"], f["description"], f["parameters"], execute)]


def oracle(row):
    expression = ast.parse(row["ground_truth"][0], mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.args
    ):
        raise ValueError("Unexpected publisher call")
    name = expression.func.id
    arguments = {kw.arg: ast.literal_eval(kw.value) for kw in expression.keywords}
    return {"name": name, "arguments": arguments}, FUNCTIONS[name](**arguments)


def score(row, trajectory):
    gold, result = oracle(row)
    calls = [t["call"] for t in trajectory["turns"] if "observation" in t]
    correct_call = calls == [gold]
    text = (trajectory["answer"] or "").strip()
    number = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", text)
    correct = bool(
        number and math.isclose(float(text), result, rel_tol=0.005, abs_tol=1e-10)
    )
    return dict(
        success=correct_call and correct and trajectory["status"] == "answered",
        path=correct_call,
        correct_answer=correct,
        completed=trajectory["status"] == "answered",
    )
