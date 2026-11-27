"""Re-execute the held calculator programs and separate input selection from computation."""

import ast
from collections import Counter
from fractions import Fraction
from pathlib import Path

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv
from scripts.learner_prefix_panel import PANEL
from scripts.register_learner_prefix import PRIMARY, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed


def quantities(expressions):
    """Literal magnitudes, including numeric quotients used to spell source decimals."""
    values = set()
    for expr in expressions:
        for node in ast.walk(ast.parse(expr, mode="eval")):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                values.add(abs(Fraction(ast.get_source_segment(expr, node))))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if all(
                    isinstance(part, ast.Constant) for part in (node.left, node.right)
                ):
                    a, b = [
                        Fraction(ast.get_source_segment(expr, p))
                        for p in (node.left, node.right)
                    ]
                    if b:
                        values.add(abs(a / b))
    return values


def run(cohort="learner-prefix"):
    panel_path, stage = PANEL, PRIMARY
    if cohort == "bounded-read":
        from scripts.register_bounded_read import PANEL as panel_path, PRIMARY as stage
    root = RESULT / stage
    reading = Path("docs/measurements") / (cohort + "-reading.json")
    for p in (panel_path, root / "checksums.json", reading):
        committed(p)
    panel = {r["id"]: r for r in unpack(panel_path)["rows"] if r["family"] == "report"}
    rows = []
    totals = {}
    for arm in ("candidate", "control"):
        stats = Counter()
        for row in unpack(root / (arm + ".json.gz")):
            if row["family"] != "report":
                continue
            w = panel[row["id"]]
            q = row["requests"][0]
            events = executions(q["trajectory"])
            current = []
            for e in events:
                xs = e["call"]["arguments"]["steps"]
                try:
                    observation = calculate(xs)
                except (ValueError, ArithmeticError, SyntaxError, KeyError) as err:
                    observation = {"error": type(err).__name__}
                assert observation == e["observation"]
                if "value" in observation:
                    current.append((xs, Fraction(observation["value"])))
            # Derive source magnitudes from expert leaves, excluding only an annotated mean divisor
            # and the separate percent-conversion step. Presence does not prove use in the right role.
            expr = w["steps"][0]
            root_expr = ast.parse(expr.strip(), mode="eval").body
            if (
                "average" in w["question"].lower()
                and isinstance(root_expr, ast.BinOp)
                and isinstance(root_expr.op, ast.Div)
                and isinstance(root_expr.right, ast.Constant)
            ):
                root_expr = root_expr.left
            expected = {
                abs(Fraction(ast.get_source_segment(expr.strip(), n)))
                for n in ast.walk(root_expr)
                if isinstance(n, ast.Constant)
            }
            assert expected
            available = quantities(current[-1][0]) if current else set()
            in_range = bool(
                current
                and abs(current[-1][1] - Fraction(w["gold"]))
                <= Fraction(w["tolerance"])
            )
            r = dict(
                id=w["id"],
                arm=arm,
                attempts=len(events),
                valid=len(current),
                answered=q["score"]["completed"],
                operands=" ".join(map(str, sorted(expected))),
                matched=" ".join(map(str, sorted(expected & available))),
                all_magnitudes=bool(current and expected <= available),
                correct_last_value=in_range,
                delivered_strict=q["score"]["success"],
                answer=q["trajectory"]["answer"] or "",
            )
            for k, v in dict(
                requests=1,
                attempted=bool(events),
                successful_call=bool(current),
                all_magnitudes=r["all_magnitudes"],
                correct_last_value=in_range,
                delivered_strict=r["delivered_strict"],
                completed=r["answered"],
            ).items():
                stats[k] += v
            rows.append(r)
        totals[arm] = dict(stats)
    write_csv(Path("paper") / (cohort + "-calculator-audit.csv"), rows)
    persist(
        Path("docs/measurements") / (cohort + "-calculator-audit.json"),
        dict(
            archive=digest(root / "checksums.json"),
            reading=digest(reading),
            totals=totals,
            method="Descriptive analysis after unmasking. Replay every observed call including errors. Count presence of all reference numerical magnitudes in the last successful call, allowing literal decimal or rational spellings. This is permissive: matching the digits does not establish the correct row, role, sign or operation. Check the last returned value separately using the registered tolerance.",
        ),
    )
    print(totals)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort", choices=("learner-prefix", "bounded-read"), default="learner-prefix"
    )
    run(p.parse_args().cohort)
