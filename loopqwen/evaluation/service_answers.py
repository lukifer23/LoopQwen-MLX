"""Operational grading for numerical file properties and observed package versions."""

import re
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w]|\.\d)")
GROUPED = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w,])")


def visible_reference(answer, resource, gold, *, file=False):
    """Ignore the caller's resource name and conventional thousands separators."""
    text = (answer or "").replace(resource, "").replace(Path(resource).name, "")
    if file:
        text = GROUPED.sub(lambda m: m[0].replace(",", ""), text)
    if gold == "package is not installed":
        return "not installed" in text.casefold() and not re.search(
            r"\bversion\s+(?:is\s+)?\d", text, re.I
        )
    return set(NUMBER.findall(text)) == {str(gold)}


def years(answer, expected):
    """Read an unambiguous age or duration, including ordinary spelled-out integers."""
    from ..arithmetic_actions import WORDS

    units = {w: n for w, n in WORDS.items() if isinstance(n, int) and n < 20}
    tens = dict(
        zip(
            "twenty thirty forty fifty sixty seventy eighty ninety".split(),
            range(20, 100, 10),
            strict=True,
        )
    )
    small = (
        r"(?:"
        + "|".join(tens)
        + r")(?:[ -](?:"
        + "|".join(w for w, v in units.items() if 0 < v < 10)
        + r"))?|"
        + "|".join(units)
    )
    group = (
        r"(?:(?:"
        + "|".join(w for w, v in units.items() if 0 < v < 10)
        + r")[ -]hundred(?:[ -](?:and[ -])?(?:"
        + small
        + r"))?|(?:"
        + small
        + r"))"
    )
    pattern = r"\b" + group + r"(?:[ -]thousand(?:[ -](?:and[ -])?" + group + r")?)?\b"

    def convert(match):
        total = part = 0
        for word in re.split(r"[ -]+", match[0].casefold()):
            if word == "hundred":
                part *= 100
            elif word == "thousand":
                total, part = total + 1000 * part, 0
            else:
                part += units.get(word, tens.get(word, 0))
        return str(total + part)

    text = re.sub(pattern, convert, answer or "", flags=re.I)
    # Spelling out the correct number with the wrong unit cannot answer a question in years.
    if re.search(r"\d[ -]*(?:months?|weeks?|days?|hours?|minutes?)\b", text, re.I):
        return False
    if re.search(r"(?<!\w)[−-]\d", text):
        return False
    return set(NUMBER.findall(text)) == {str(expected)}
