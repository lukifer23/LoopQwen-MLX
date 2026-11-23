"""Strict V2 scalar answer protocol, independent of MLX and model assets."""

import re
from fractions import Fraction

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
MARKER = r"(?:final answer\s*:\s*|answer\s*:\s*|the answer is\s+|####\s*)"


def numeric(text):
    text = text.split("<|endoftext|>")[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    marked = [i for i, s in enumerate(lines) if re.match(MARKER, s, re.I)]
    if marked and marked[-1] != len(lines) - 1:
        return None
    candidates = [lines[i] for i in marked] if marked else lines
    vals = []
    for line in candidates:
        line = re.sub("^" + MARKER, "", line, flags=re.I).strip().strip("$")
        line = re.sub(r"^\\boxed\{(.*)\}$", r"\1", line)
        line = re.sub(r"^\\frac\{(-?\d+)\}\{(\d+)\}$", r"\1/\2", line)
        m = re.fullmatch(rf"({NUMBER})(?:\s*/\s*({NUMBER}))?[.!]?", line)
        if not m or any(abs(int(e)) > 512 for e in re.findall(r"[eE]([+-]?\d+)", line)):
            return None
        try:
            n, d = (
                Fraction(x.replace(",", "")) if x else Fraction(1) for x in m.groups()
            )
            vals.append(str(n / d))
        except (ZeroDivisionError, ValueError):
            return None
    return vals[0] if len(set(vals)) == 1 else None
