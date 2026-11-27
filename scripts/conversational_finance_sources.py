"""Rebuild original multi-turn report questions and verify every numeric program against a real tool."""

import argparse
import json
import re
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from scripts.content_read_material import persist_rows
from scripts.report_calculation_material import OPERATORS
from scripts.shared_query_material import numeric_argument
from scripts.table_policy_material import persist

ROOT = Path("research/sources/conversational-finance")
FILE = Path("data/research_external/convfinqa/data.zip")
REV = "cf3eed2d5984960bf06bb8145bcea5e80b0222a6"
URL = f"https://raw.githubusercontent.com/czyssrs/ConvFinQA/{REV}/data.zip"
SHA = "d764271fae60d81b62e6d58dfc481807ebc8cfbcd633811241723c4a2101072a"


def convert(program):
    ops = re.findall(r"(\w+)\(([^()]+), ([^()]+)\)", program)
    if not ops:
        return [numeric_argument(program)], False
    if ", ".join(f"{o}({a}, {b})" for o, a, b in ops) != program or any(
        o not in OPERATORS for o, _, _ in ops
    ):
        raise ValueError("Unsupported released program")
    return [
        f"({numeric_argument(a)}){OPERATORS[o]}({numeric_argument(b)})"
        for o, a, b in ops
    ], True


def build(download):
    if download and (not FILE.exists() or digest(FILE) != SHA):
        FILE.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as f:
            data = f.read()
        assert sha256(data).hexdigest() == SHA
        FILE.write_bytes(data)
    assert digest(FILE) == SHA
    with ZipFile(FILE) as z:
        original = z.read("data/train.json")
        conv = json.loads(original)
        dev = {x["filename"] for x in json.loads(z.read("data/dev.json"))}
        private = {x["filename"] for x in json.loads(z.read("data/test_private.json"))}
    reserved = set(dev) | private
    sources = {}
    for split in ("dev", "test"):
        path = Path("data/research_external/finqa") / (split + ".json")
        sources[str(path)] = digest(path)
        reserved.update(x["filename"] for x in read(path))
    reserved_issuers = set(
        read("research/data/report-calculation-panel.json")["held_out_issuers"]
    )
    for path in (
        Path("research/sources/shared-query/held.json.gz"),
        Path("research/sources/repair-query/transfer-panel.json.gz"),
    ):
        sources[str(path)] = digest(path)
        values = unpack(path)
        for row in values if isinstance(values, list) else values["rows"]:
            if row.get("family") == "report":
                reserved.add(row["source"])
                reserved_issuers.add(row["cluster"])
    good = []
    for doc in conv:
        name = doc["filename"]
        issuer = name.split("/")[0]
        if name in reserved or issuer in reserved_issuers:
            continue
        a = doc["annotation"]
        questions = a["dialogue_break"]
        if not 3 <= len(questions) <= 6 or any(not q.strip() for q in questions):
            continue
        chain = False
        turns = []
        try:
            for q, prog, published in zip(
                questions, a["turn_program"], a["exe_ans_list"], strict=True
            ):
                steps, math = convert(prog)
                result = calculate(steps)
                value = Fraction(result["value"])
                gold = Fraction(str(published))
                if abs(value - gold) > Fraction(51, 10**7):
                    raise ValueError(
                        "Exact rational answer disagrees with released rounded execution"
                    )
                chain |= "#0" in prog and any(t["arithmetic"] for t in turns)
                turns.append(
                    dict(
                        question=q,
                        program=prog,
                        arithmetic=math,
                        steps=steps,
                        result=result,
                        release_value=str(published),
                    )
                )
        except (ValueError, ArithmeticError, SyntaxError, KeyError, TypeError):
            continue
        if not chain:
            continue
        body = "\n".join(
            [
                *doc["pre_text"],
                "\n".join(" | ".join(row) for row in doc["table"]),
                *doc["post_text"],
            ]
        )
        good.append(
            dict(
                id=doc["id"],
                file=name,
                issuer=issuer,
                document=body,
                turns=turns,
                evidence={
                    key: doc[key]["gold_inds"]
                    for key in ("qa", "qa_0", "qa_1")
                    if key in doc
                },
            )
        )
    order = sorted(good, key=lambda w: fingerprint([677196, w["id"]]))
    selected = []
    by = Counter()
    pages = set()
    for w in order:
        if w["file"] in pages or by[w["issuer"]] >= 4:
            continue
        selected.append(w)
        pages.add(w["file"])
        by[w["issuer"]] += 1
        if len(selected) == 96:
            break
    assert (
        len(selected) >= 32
        and len(by) >= 12
        and not reserved.intersection(pages)
        and not reserved_issuers.intersection(by)
    )
    ROOT.mkdir(exist_ok=True, parents=True)
    dest = ROOT / "training.json.gz"
    persist_rows(dest, selected)
    receipt = dict(
        url=URL,
        revision=REV,
        zip=SHA,
        train_json=sha256(original).hexdigest(),
        split="publisher training",
        records=digest(dest),
        previous_material=sources,
        report_panel=digest(Path("research/data/report-calculation-panel.json")),
        original=len(conv),
        eligible=len(good),
        selected=len(selected),
        issuers=len(by),
        pages=len(pages),
        questions=sum(len(w["turns"]) for w in selected),
        arithmetic=sum(t["arithmetic"] for w in selected for t in w["turns"]),
        reserved_pages=len(reserved),
        reserved_project_issuers=sorted(reserved_issuers),
        method="Original published complete report text/table and ordered conversational questions. Translate only literal lookups and four exact arithmetic operations with backward references; execute each through the public calculator and compare the rational return to the released rounded answer, absolute tolerance 0.0000051. Require 3–6 turns and a later calculation referencing a sub-result after a preceding arithmetic turn. Exclude every ConvFinQA dev/private-test page, all FinQA dev/test pages and all report issuers in the identified earlier project panels. Deterministic hash order, at most one conversation per page and four per issuer. Lookup replay validates a numeric annotation; it is not evidence a model should call a calculator to read a cell. Source qualification only, no model performance measured. Published programs can describe equivalent quantities at a different display scale than the human question; keep the text and every program for source review before training.",
    )
    persist(ROOT / "inventory.json", receipt)
    print(
        {
            k: receipt[k]
            for k in ("eligible", "selected", "issuers", "questions", "arithmetic")
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    build(p.parse_args().download)
