"""Build full-report and table questions with executed supervision between calculation steps."""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.calculator_tool import calculate, tool
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import compact, initial_messages, prompt

DATA = Path("research/data")
SEED = 621807
FIN = "0f16e2867befa6840783e58be38c9efb9229d742"
TAB = "5a1a52214521590b075f545b76a4f5ce666345e3"
OPERATORS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
INSTRUCTION = "Use the available calculator when useful. Give the final number in the units requested. Report percentages in percentage points (25 for 25%)."


def load(cache, name, url):
    path = cache / name
    if not path.is_file():
        path.write_bytes(urllib.request.urlopen(url, timeout=90).read())
    return read(path), {"url": url, "sha256": digest(path)}


def scalar(raw):
    text = str(raw).strip().replace(",", "").replace("$", "").rstrip("%").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError("Non-scalar human reference")
    return Decimal(text)


def finance(row):
    qa = row["qa"]
    target = scalar(qa["answer"])
    tokens = re.findall(r"(\w+)\(([^()]+), ([^()]+)\)", qa["program"])
    if (
        not 1 <= len(tokens) <= 4
        or ", ".join(f"{op}({l}, {r})" for op, l, r in tokens) != qa["program"]
    ):
        raise ValueError("Unsupported expert program")
    steps = []
    values = []

    def literal(token):
        if token.startswith("#"):
            ref = int(token[1:])
            if not 0 <= ref < len(values):
                raise ValueError("Unknown reference")
            return str(values[ref])
        text = token.removeprefix("const_")
        if token.startswith("const_m"):
            text = "-" + token[7:]
        v = Fraction(scalar(text))
        if text.endswith("%"):
            v /= 100
        return str(v)

    for op, l, r in tokens:
        expr = f"({literal(l)}){OPERATORS[op]}({literal(r)})"
        v = Fraction(calculate([expr])["value"])
        values.append(v)
        steps.append(expr)
    if abs(float(values[-1]) - float(qa["exe_ans"])) > max(
        0.00002, abs(float(qa["exe_ans"])) * 0.000001
    ):
        raise ValueError("Reconstruction disagrees with released execution")
    tol = Decimal(1).scaleb(min(0, target.as_tuple().exponent)) / 2
    result = Decimal(values[-1].numerator) / Decimal(values[-1].denominator)
    if (
        "%" in str(qa["answer"])
        and abs(100 * result - target) <= tol
        and abs(result - target) > tol
    ):
        steps.append(f"({values[-1]})*100")
        result *= 100
    if abs(result - target) > tol or len(steps) > 4:
        raise ValueError("Expert program and human numeric answer disagree")
    if result == target:
        tol = Decimal(0)
    body = "\n".join(
        [
            *row["pre_text"],
            "\n".join(" | ".join(line) for line in row["table"]),
            *row["post_text"],
        ]
    )
    return {
        "id": row["id"],
        "source": "FinQA",
        "cluster": row["filename"].split("/")[0],
        "report": row["filename"],
        "context": body,
        "question": qa["question"],
        "gold": str(target),
        "tolerance": str(tol),
        "steps": steps,
        "evidence": list(qa["gold_inds"].values()),
    }


def request(w):
    return (
        (w["context"] + "\n\n" if w.get("context") else "")
        + w["question"]
        + "\n"
        + INSTRUCTION
    )


def exchanges(w, tok):
    msgs = initial_messages(request(w), [tool()])
    examples = []
    for i, expression in enumerate(w["steps"]):
        args = {"steps": [expression]}
        visible = compact({"name": "calculate", "arguments": args})
        thinking = w["reasons"][i] if "reasons" in w else ""
        if "evidence" in w and w["evidence"] and i == 0:
            evidence = w["evidence"][0]
            thinking = tok.decode(tok.encode(evidence, add_special_tokens=False)[:48])
        examples.append(
            {
                "id": w["id"] + f"-step-{i}",
                "messages": list(msgs),
                "visible": visible,
                "thinking": thinking,
            }
        )
        msgs.extend(
            [
                {"role": "assistant", "content": visible},
                {"role": "tool", "content": compact(calculate(**args))},
            ]
        )
    examples.append(
        {
            "id": w["id"] + "-final",
            "messages": msgs,
            "visible": w["gold"],
            "thinking": "",
        }
    )
    return examples


def gsm(row):
    q = row["question"]
    body, answer = row["answer"].rsplit("####", 1)
    target = scalar(answer)
    found = list(re.finditer(r"<<([^<>=]+)=([^<>]+)>>", body))
    if not 2 <= len(found) <= 4:
        raise ValueError("Require two to four explicit computations")
    steps = []
    reasons = []
    start = 0
    for m in found:
        expr = m[1].replace(",", "").replace("×", "*")
        actual = Fraction(calculate([expr])["value"])
        if actual != Fraction(scalar(m[2])):
            raise ValueError("Changed human operation")
        # Use the new line leading up to the annotation, excluding the result itself.
        line = body[start : m.start()].split("\n")[-1].strip()
        reasons.append(line)
        steps.append(expr)
        start = m.end()
    if actual != Fraction(target):
        raise ValueError("Last action does not return the human result")
    return {
        "id": "gsm-" + hashlib.sha256(q.encode()).hexdigest(),
        "question": q,
        "gold": str(target),
        "steps": steps,
        "reasons": reasons,
    }


def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tok = load_tokenizer(local_snapshot())
    url = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FIN}/dataset/"
    dev, dev_src = load(cache, "finqa-dev.json", url + "dev.json")
    train, train_src = load(cache, "train.json", url + "train.json")
    tab, tab_src = load(
        cache,
        "tabmwp.json",
        f"https://raw.githubusercontent.com/lupantech/PromptPG/{TAB}/data/tabmwp/problems_test.json",
    )
    panel = []
    issuers = set()
    tables = set()
    rng.shuffle(dev)
    for r in dev:
        try:
            w = finance(r)
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            ZeroDivisionError,
            InvalidOperation,
        ):
            continue
        if w["cluster"] in issuers:
            continue
        issuers.add(w["cluster"])
        panel.append(w)
        if len(issuers) == 24:
            break
    items = list(tab.items())
    rng.shuffle(items)
    for identifier, r in items:
        if r["ques_type"] != "free_text" or r["ans_type"] not in (
            "integer_number",
            "decimal_number",
        ):
            continue
        try:
            target = scalar(r["answer"])
        except ValueError:
            continue
        table = r["table"]
        h = hashlib.sha256(table.encode()).hexdigest()
        if h in tables:
            continue
        tables.add(h)
        panel.append(
            {
                "id": "tab-" + identifier,
                "cluster": h,
                "source": "TabMWP",
                "context": (r["table_title"] + "\n" if r.get("table_title") else "")
                + table,
                "question": r["question"],
                "gold": str(target),
                "tolerance": "0",
                "unit": r.get("unit", ""),
            }
        )
        if len(tables) == 24:
            break
    assert len(panel) == 48
    fit = []
    fit_rows = []
    rng.shuffle(train)
    for r in train:
        try:
            w = finance(r)
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            ZeroDivisionError,
            InvalidOperation,
        ):
            continue
        if w["cluster"] in issuers or len(tok.encode(request(w))) > 800:
            continue
        ex = exchanges(w, tok)
        fit += ex
        fit_rows.append(w)
        if len(fit_rows) == 256:
            break
    gpath = Path("data/research_external/gsm-train.jsonl")
    grand = [json.loads(s) for s in gpath.read_text().splitlines()]
    rng.shuffle(grand)
    for r in grand:
        try:
            w = gsm(r)
        except (ValueError, SyntaxError, ZeroDivisionError, InvalidOperation):
            continue
        fit += exchanges(w, tok)
        fit_rows.append({**w, "source": "GSM8K"})
        if len(fit_rows) == 640:
            break
    assert len(fit_rows) == 640 and sorted(
        Counter(w["source"] for w in fit_rows).values()
    ) == [256, 384]
    old = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    rng.shuffle(old)
    fit += old[:192]
    rng.shuffle(panel)
    for w in panel:
        w["user_request"] = request(w)
        w["initial_ids"] = tok.encode(
            prompt(tok, initial_messages(w["user_request"], [tool()])),
            add_special_tokens=False,
        )
    sources = {
        "finqa_train": train_src,
        "finqa_dev": dev_src,
        "tabmwp_test": tab_src,
        "gsm_train_sha256": digest(gpath),
    }
    save(
        DATA / "report-calculation-training.json",
        {"sources": sources, "episodes": fit_rows, "examples": fit},
    )
    save(
        DATA / "report-calculation-panel.json",
        {"sources": sources, "held_out_issuers": sorted(issuers), "rows": panel},
    )
    print(
        len(fit),
        "response targets;",
        sum(
            len(tok.encode(prompt(tok, w["messages"])))
            + len(tok.encode(w["visible"] + w["thinking"]))
            for w in fit
        ),
        "fitting tokens;",
        min(len(w["initial_ids"]) for w in panel),
        max(len(w["initial_ids"]) for w in panel),
        "test prompt range",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    main(p.parse_args().cache)
