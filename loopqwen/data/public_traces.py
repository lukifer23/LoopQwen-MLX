"""Optional publisher solutions with checked calculator annotations and final marker."""

import ast
from fractions import Fraction
import hashlib
import json
import operator
import re
from urllib.request import urlopen
from transformers import AutoTokenizer
from .build_data import GSM_REV
from .v2 import ROOT
from ..model_loader import local_snapshot


def value(node):
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -value(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](value(node.left), value(node.right))
    raise ValueError("Unsupported calculator expression")


def gsm(cap):
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/"
    path = ROOT / "gsm_source.jsonl"
    if not path.exists():
        license_text = urlopen(url + "LICENSE", timeout=30).read()
        if not license_text.startswith(b"MIT License"):
            raise ValueError("Unexpected license")
        (ROOT / "GSM-LICENSE").write_bytes(license_text)
        raw = urlopen(url + "grade_school_math/data/train.jsonl", timeout=30).read()
        path.write_bytes(raw)
    tok = AutoTokenizer.from_pretrained(str(local_snapshot()))
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        item = json.loads(line)
        solution = item["answer"]
        steps = re.findall(r"<<([^<>]+)=([^<>=]+)>>", solution)
        if not steps:
            continue
        try:
            if not all(
                value(ast.parse(expr.replace(",", ""), mode="eval").body)
                == Fraction(total.replace(",", ""))
                for expr, total in steps
            ):
                continue
            gold = Fraction(solution.split("####")[-1].strip().replace(",", ""))
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        prompt = (
            item["question"]
            + "\nShow your steps. End with Final answer: <number>.\nSolution:"
        )
        ids = tok.encode(prompt, add_special_tokens=False)
        p = len(ids)
        ids += tok.encode(
            " " + solution.replace("####", "Final answer:"), add_special_tokens=False
        ) + [tok.eos_token_id]
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if len(ids) - 1 > cap or key in seen:
            continue
        seen.add(key)
        rows.append(
            dict(
                id=key,
                ids=ids,
                prefix=p,
                prompt=prompt,
                answer=str(gold),
                steps=len(steps),
                complexity=0,
                category="gsm8k",
                protocol="gsm",
                input_length=len(ids) - 1,
                output_length=len(ids) - p,
            )
        )
    return rows, dict(
        url=url,
        revision=GSM_REV,
        license="MIT",
        eligible=len(rows),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
