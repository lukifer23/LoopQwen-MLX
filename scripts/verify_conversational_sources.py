"""Verify exact conversational arithmetic from the publisher's original operands independently."""

import json
import re
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from scripts.conversational_finance_sources import FILE, ROOT, SHA
from scripts.conversational_teacher import PLAN, assessment, select
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("docs/measurements/conversational-finance-sources.json")


def execute(program):
    previous = []

    def number(x):
        if x.startswith("#"):
            return previous[int(x[1:])]
        s = x.removeprefix("const_").replace(",", "").replace("$", "")
        if s.startswith("m"):
            s = "-" + s[1:]
        scale = 100 if s.endswith("%") else 1
        return Fraction(s.rstrip("%")) / scale

    for match in re.finditer(
        r"(add|subtract|multiply|divide)\(([^()]+), ([^()]+)\)", program
    ):
        a, b = number(match[2]), number(match[3])
        if match[1] == "add":
            v = a + b
        elif match[1] == "subtract":
            v = a - b
        elif match[1] == "multiply":
            v = a * b
        else:
            v = a / b
        previous.append(v)
    return previous or [number(program)]


def run():
    for p in (ROOT / "inventory.json", ROOT / "training.json.gz", PLAN):
        committed(p)
    assert digest(FILE) == SHA
    with ZipFile(FILE) as z:
        raw = {r["id"]: r for r in json.loads(z.read("data/train.json"))}
    rows = unpack(ROOT / "training.json.gz")
    n = 0
    for w in rows:
        d = raw[w["id"]]
        qs = d["annotation"]
        assert w["file"] == d["filename"] and len(w["turns"]) == len(
            qs["dialogue_break"]
        )
        for i, t in enumerate(w["turns"]):
            assert (
                t["question"] == qs["dialogue_break"][i]
                and t["program"] == qs["turn_program"][i]
            )
            expected = execute(t["program"])
            assert list(map(Fraction, t["result"]["results"])) == expected
            assert expected[-1] == Fraction(t["result"]["value"])
            assert t["result"] == calculate(t["steps"])
            n += 1
    qualified = []
    for w, i in select():
        t = w["turns"][i]

        def record(steps):
            reply = dict(
                turns=[
                    dict(
                        call=dict(name="calculate", arguments=dict(steps=steps)),
                        observation=calculate(steps),
                    )
                ]
            )
            return assessment(w, i, reply)

        gold = record(t["steps"])
        literal = (
            record([t["result"]["value"]]) if "/" not in t["result"]["value"] else None
        )
        assert not literal or not literal["accepted"]
        qualified.append(
            dict(
                id=w["id"],
                turn=i,
                official_passed=gold["accepted"],
                literal_control=literal,
            )
        )
    out = dict(
        material=digest(ROOT / "training.json.gz"),
        inventory=digest(ROOT / "inventory.json"),
        plan=digest(PLAN),
        publisher_questions=n,
        original_documents=len(rows),
        issuer_count=len({x["issuer"] for x in rows}),
        official_training_teacher_control=sum(v["official_passed"] for v in qualified),
        controls=qualified,
    )
    persist(OUT, out)
    print(n, out["official_training_teacher_control"])


if __name__ == "__main__":
    run()
