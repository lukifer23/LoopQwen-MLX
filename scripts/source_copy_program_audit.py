"""Retrospective source-operand sensitivity of the archived financial conversation calls."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, digest
from loopqwen.research.native_function_study import executions
from loopqwen.research.source_copy_transfer import SCALES, SECOND_AMOUNTS
from scripts.assistance_deliberation import write_csv
from scripts.register_source_copy import RESULT, PAIR
from scripts.shared_query_calculation_replay import expert, execute_change, SHIFTS
from scripts.source_copy_material import PANEL
from scripts.table_policy_material import persist

OUT = Path("docs/measurements/source-copy-program-audit.json")


def check(w, steps, ratio):
    target, operands = expert(w)
    assert target == Fraction(w["observed"]["value"])
    original = Fraction(calculate(steps)["value"])
    # Percent answers may be computed as a ratio or percentage-points value.
    # A single normalization must agree on the unmodified and every perturbed input.
    scales = [s for s in ([1, 100] if ratio else [1]) if original == s * target]
    tests = []
    for operand in sorted(operands):
        for delta in SHIFTS:
            expected, _ = expert(w, operand, delta)
            if expected == target:
                continue
            try:
                computed = execute_change(steps, operand, delta)
            except ZeroDivisionError:
                computed = None
            scales = [s for s in scales if computed == s * expected]
            tests.append(
                dict(
                    operand=str(operand),
                    delta=str(delta),
                    expected=str(expected),
                    actual=None if computed is None else str(computed),
                )
            )
    return dict(
        original=str(original),
        correct_value=original == target or (ratio and original == 100 * target),
        eligible=bool(tests),
        matches=bool(tests and scales),
        tests=tests,
    )


def main():
    root = RESULT / PAIR
    docs = {w["id"]: w for w in unpack(PANEL) if w["family"] == "report"}
    controls = []
    for w in docs.values():
        for i, t in enumerate(w["turns"]):
            ratio = (
                SCALES[w["cluster"]] is None
                if i == 0
                else w["cluster"] not in SECOND_AMOUNTS
            )
            good = check(t, t["steps"], ratio)
            val = Fraction(t["observed"]["value"])
            pasted = [f"{val.numerator*1000003}/{val.denominator*1000003}"]
            sham = check(t, pasted, ratio)
            assert (good["matches"] or not good["eligible"]) and not sham["matches"]
            controls.append(
                dict(
                    id=w["id"],
                    turn=i,
                    eligible=good["eligible"],
                    tests=len(good["tests"]),
                )
            )
    rows = []
    details = []
    sums = {}
    for arm in ("candidate", "control"):
        counts = Counter()
        for episode in unpack(root / (arm + ".json.gz")):
            if episode["id"] not in docs:
                continue
            w = docs[episode["id"]]
            for i, req in enumerate(episode["requests"]):
                t = w["turns"][min(i, 1)]
                ratio = (
                    SCALES[w["cluster"]] is None
                    if i == 0
                    else w["cluster"] not in SECOND_AMOUNTS
                )
                calls = executions(req["trajectory"])
                valid = [c for c in calls if "value" in c["observation"]]
                checks = [
                    check(t, c["call"]["arguments"]["steps"], ratio) for c in valid
                ]
                last = checks[-1] if checks else {}
                flags = dict(
                    requests=True,
                    executed=bool(calls),
                    valid=bool(valid),
                    any_correct_return=any(c["correct_value"] for c in checks),
                    any_operative=any(c["matches"] for c in checks),
                    last_correct_return=last.get("correct_value", False),
                    last_operative=last.get("matches", False),
                    strict_delivery=bool(req["score"]["success"]),
                    last_operative_and_delivered=last.get("matches", False)
                    and bool(req["score"]["success"]),
                )
                counts.update({k: int(v) for k, v in flags.items()})
                identity = dict(arm=arm, id=w["id"], presentation=req["history"])
                rows.append(dict(**identity, **flags))
                details.append(dict(**identity, checks=checks))
        sums[arm] = dict(counts)
    write_csv(Path("paper/source-copy-program-audit.csv"), rows)
    persist(
        OUT,
        dict(
            primary=digest(root / "checksums.json"),
            review=digest(Path("docs/measurements/source-copy-reading/report.json")),
            method="Post hoc. Hold each original question, source, model output and final successful call fixed. At each annotated nonconstant operand, apply +11/7 and -5/13 to that value in the published program and the emitted program and replay. Require equality at the original input and all operative changes. Allow a consistent factor of 100 for a percentage result. Literal decimals and reduced fractions are recognized. This diagnostic measures source-value sensitivity within one call. Programs using separately computed receipts or alternative unit conversions may fail. Publisher constants alone provide no perturbable source. Previous frozen scores are retained.",
            controls=controls,
            results=sums,
            details=details,
        ),
    )
    print(sums)


if __name__ == "__main__":
    main()
