import re
from collections import defaultdict
from fractions import Fraction

from .generation_eval import generate

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"


def normalized(text):
    text = text.split("<|endoftext|>")[0].strip()
    if not text:
        return None
    line = text.splitlines()[0].strip().strip("$")
    line = re.sub(r"^\\boxed\{([^{}]+)\}$", r"\1", line)
    line = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", line)
    match = re.fullmatch(
        r"(?:the answer is\s*|answer:\s*)?(" + NUMBER + r")[.!]?\s*", line, re.I
    )
    if not match:
        return None
    try:
        return str(Fraction(re.sub(r"[,\s]", "", match[1])))
    except ZeroDivisionError:
        return None


def math_eval(model, tok, rows, max_tokens, matched=False):
    records = []
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        # Match the ceiling on decoder-layer executions including prompt processing and
        # the prefill-produced first output token. Floor the dense allowance.
        prefix = row["prefix"]
        budget = (
            ((prefix + max_tokens - 1) * (24 + model.span.length) // 24 - prefix + 1)
            if matched
            else max_tokens
        )
        gen = generate(model, tok, row["prompt"], budget)
        pred = normalized(gen["text"])
        target = normalized(row["answer"])
        if target is None:
            raise ValueError(f"Unscorable gold {row['answer']}")
        ok = pred == target
        for key in (
            "all",
            row["category"],
            f"steps-{row['steps']}",
            "prompt-1..20"
            if prefix <= 20
            else "prompt-21..40"
            if prefix <= 40
            else "prompt-41+",
        ):
            groups[key][0] += int(ok)
            groups[key][1] += 1
        records.append(
            dict(
                id=row["id"],
                category=row["category"],
                steps=row["steps"],
                gold=row["answer"],
                prediction=pred,
                correct=ok,
                layer_executions=len(model.sites)
                * (prefix + len(gen["token_ids"]) - 1),
                **gen,
            )
        )
    return dict(
        groups={
            k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1])
            for k, v in groups.items()
        },
        records=records,
    )
