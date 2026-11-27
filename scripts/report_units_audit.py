"""Recount final report quantities in physical units whenever the answer states a scale."""

import re
from fractions import Fraction

from loopqwen.evaluation.answers import NUMBER, numeric

SCALE = {
    "": 1,
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "percent": Fraction(1, 100),
}
PRINTED = re.compile(rf"(?<![\w.])({NUMBER})(?:\s*/\s*({NUMBER}))?")
NAMED = re.compile(r"\s*(thousand|million|billion|percent)\b|\s*(%)", re.I)
BASE_UNIT = re.compile(
    r"\s*(?:US\s+)?(?:dollars?|shares?|euros?|pounds?|GBP|USD|EUR)\b", re.I
)


def visible_scale(text, source_scale, gold):
    if not text:
        return False, "no visible answer"
    text = text.replace("**", "").replace("−", "-").strip()
    # Use the same terminal-quantity convention as the frozen report score.
    matches = list(PRINTED.finditer(text))
    if not matches:
        return False, "no numeral"
    m = matches[-1]
    value = numeric(m[0])
    if value is None:
        return False, "invalid numeral"
    amount = Fraction(value)
    prefix, suffix = text[: m.start()].rstrip(), text[m.end() :]
    unit = NAMED.match(suffix)
    if unit:
        named = (unit[1] or "percent").lower()
        actual = amount * SCALE[named]
        label = named
    elif BASE_UNIT.match(suffix) or prefix.endswith(("$", "£", "€")):
        actual = amount
        label = "unscaled explicit base unit"
    else:
        actual = amount * SCALE[source_scale]
        label = "implicit source scale"
    return actual == Fraction(gold) * SCALE[source_scale], label
