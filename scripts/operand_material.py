"""Reserve disjoint human math problems and verify each training program against its answer."""

import ast
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from loopqwen.arithmetic_actions import CONSTANTS, OPS, as_steps, mentions
from loopqwen.calculator_tool import execute
from loopqwen.evaluation.answers import numeric
from loopqwen.io import digest, read, save
from program_reasoning_material import (
    ASDIV_SHA,
    ASDIV_URL,
    former,
    norm,
    program,
    sha,
    source,
)

DATA = Path("research/data")
MAWPS = "https://raw.githubusercontent.com/arkilpatel/SVAMP/78e727689e1c1bebfc4be39c446898e8e10b0518/data/cv_mawps/fold0/train.csv"
MSHA = "0674253e6099813cefbb1d6eae3d2dc784ab6ebfdaf8755f6ebb4b99fc1f6ebe"


def grams(text):
    ws = ["#" if any(c.isdigit() for c in w) else w for w in norm(text)]
    return {tuple(ws[i : i + 4]) for i in range(len(ws) - 3)}


def external():
    ad = source("asdiv.xml", ASDIV_URL, ASDIV_SHA)
    sv = read("research/experiments/checked-rationale-01.json")["sources"]["svamp"]
    sw = source(sv["file"], sv["url"], sv["sha256"])
    rows = [
        (
            "SVAMP",
            r["Body"].strip() + " " + r["Question"].strip(),
            numeric(str(r["Answer"])),
        )
        for r in read(sw)
    ]
    openings = {tuple(norm(q)[:8]) for _, q, _ in rows}
    for r in ET.parse(ad).iter("Problem"):
        q = " ".join(
            (r.findtext("Body", "") + " " + r.findtext("Question", "")).split()
        )
        if tuple(norm(q)[:8]) not in openings:
            rows.append(
                (
                    "ASDiv",
                    q,
                    numeric(re.sub(r"\s*\([^()]*\)\s*$", "", r.findtext("Answer", ""))),
                )
            )
    return rows


def reserve(all_external):
    excluded = former({r[1] for r in all_external})
    prior = (
        read(DATA / "tool-program-reasoning-01.json")["rows"]
        + read(DATA / "release-mixed-field-01.json")["rows"]
    )
    blocked = {tuple(norm(r["question"])[:8]) for r in prior if "question" in r} | {
        tuple(norm(q)[:8]) for q in excluded
    }
    selected = []
    counts = Counter()
    for family, q, g in sorted(
        all_external, key=lambda r: sha("operand-native-927:" + r[1])
    ):
        opening = tuple(norm(q)[:8])
        if counts[family] < 24 and g is not None and opening not in blocked:
            selected.append(dict(id=sha(q), question=q, source=family, gold=g))
            counts[family] += 1
            blocked.add(opening)
    assert len(selected) == 48
    return selected


def mawps_examples(path):
    for r in csv.DictReader(path.open()):
        values = [str(Fraction(s)) for s in r["Numbers"].split()]

        def fill(m):
            return values[int(m.group()[6:])]

        q = re.sub(r"number\d+", fill, r["Question"])
        q = re.sub(r"\s+([?,.!])", r"\1", q)
        ns = [n[2] for n in mentions(q)]
        tokens = iter(r["Equation"].split())
        actions = []

        def expr():
            t = next(tokens)
            if t in OPS and len(t) == 1:
                l, r = expr(), expr()
                actions.append([t, l, r])
                return "r" + str(len(actions) - 1)
            v = values[int(t[6:])] if t.startswith("number") else str(Fraction(t))
            return "n" + str(ns.index(v)) if v in ns else "c" + str(CONSTANTS.index(v))

        try:
            expr()
            if (
                next(tokens, None) is None
                and actions
                and len(actions) <= 6
                and execute(as_steps(ns, actions))[-1] == Fraction(r["Answer"])
            ):
                yield dict(
                    id=sha(q),
                    question=q,
                    actions=actions,
                    source="MAWPS",
                    gold=str(Fraction(r["Answer"])),
                )
        except (ValueError, ZeroDivisionError, StopIteration, IndexError):
            continue


def gsm_examples(path):
    symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        q = r["question"]
        steps = program(r["answer"])
        if not steps:
            continue
        ns = [n[2] for n in mentions(q)]
        actions = []
        results = []

        def walk(e):
            if isinstance(e, ast.Name):
                return results[int(e.id[1:]) - 1]
            if isinstance(e, ast.Constant):
                v = str(Fraction(str(e.value)))
                return (
                    "n" + str(ns.index(v)) if v in ns else "c" + str(CONSTANTS.index(v))
                )
            op = symbols[type(e.op)]
            l, rr = walk(e.left), walk(e.right)
            actions.append([op, l, rr])
            return "r" + str(len(actions) - 1)

        try:
            for s in steps:
                results.append(walk(ast.parse(s, mode="eval").body))
            original = execute(steps)[-1]
            if (
                0 < len(actions) <= 6
                and results[-1] == "r" + str(len(actions) - 1)
                and execute(as_steps(ns, actions))[-1] == original
            ):
                yield dict(
                    id=sha(q),
                    question=q,
                    actions=actions,
                    source="GSM8K",
                    gold=str(original),
                )
        except (ValueError, IndexError, KeyError, AttributeError, ZeroDivisionError):
            continue


def main():
    ext = external()
    panel = reserve(ext)
    pool = [
        *mawps_examples(source("mawps-fold0-train.csv", MAWPS, MSHA)),
        *gsm_examples(Path("data/research_external/gsm-train.jsonl")),
    ]
    # Screen against entire human transfer sources, not only the chosen evaluation items.
    fixed = [grams(q) for _, q, _ in ext]
    postings = defaultdict(list)
    for i, gs in enumerate(fixed):
        for g in gs:
            postings[g].append(i)
    excluded = former({r["question"] for r in pool})
    seen = set()
    rows = []
    count = Counter()
    for r in sorted(pool, key=lambda r: sha("operand-fit-197:" + r["id"])):
        g = grams(r["question"])
        key = tuple(norm(r["question"]))
        if key in seen or r["question"] in excluded or count[r["source"]] >= 1024:
            continue
        candidates = set(i for k in g for i in postings.get(k, []))
        if any(len(g & fixed[i]) / len(g | fixed[i]) >= 0.45 for i in candidates):
            continue
        seen.add(key)
        rows.append(r)
        count[r["source"]] += 1
    assert min(count.values()) >= 512 and len(count) == 2
    save(
        DATA / "operand-planner-training.json",
        dict(
            mawps_url=MAWPS,
            mawps_sha=MSHA,
            gsm_sha=digest(Path("data/research_external/gsm-train.jsonl")),
            jaccard_limit=0.45,
            counts=dict(count),
            rows=rows,
        ),
    )
    save(DATA / "operand-planner-test.json", dict(rows=panel))
    print(count, Counter(len(r["actions"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
