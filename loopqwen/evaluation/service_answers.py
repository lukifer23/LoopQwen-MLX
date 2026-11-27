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
