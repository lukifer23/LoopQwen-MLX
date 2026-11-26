"""Represent an arithmetic program as operations on text quantities, constants and results."""

import re
from fractions import Fraction

OPS = "+-*/"
CONSTANTS = tuple(map(str, (1, 2, 3, 4, 5, 7, 10, 12, 24, 30, 60, 100)))
WORDS = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        range(21),
    )
)
WORDS.update(dozen=12, half=Fraction(1, 2), quarter=Fraction(1, 4))
NUMBER = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?!\w)|\b(?:"
    + "|".join(WORDS)
    + r")\b",
    re.I,
)


def mentions(text):
    result = []
    for m in NUMBER.finditer(text):
        raw = m.group().casefold()
        val = Fraction(WORDS[raw]) if raw in WORDS else Fraction(raw.replace(",", ""))
        result.append((m.start(), m.end(), str(val)))
    return result


def literal(value):
    return "(" + value + ")" if "/" in value else value


def as_steps(values, actions):
    def render(ref):
        kind, idx = ref[0], int(ref[1:])
        if kind == "r":
            return "r" + str(idx + 1)
        return literal((CONSTANTS if kind == "c" else values)[idx])

    return [render(left) + op + render(right) for op, left, right in actions]
