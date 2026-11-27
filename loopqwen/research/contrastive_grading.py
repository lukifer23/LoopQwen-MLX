"""Source-based automatic diagnostics used by the fitted and freely acting models."""

import re

from scripts.program_language_material import norm


def selected_choice(text, choices):
    content = norm(text)
    if not content:
        return None
    options = [(q, norm(q)) for q in choices]
    for q, word in options:
        if content in (
            word,
            "the answer is " + word,
            "the correct answer is " + word,
            "answer " + word,
        ):
            return q
    # A short direct answer may precede an explanation. Do not give keyword credit
    # for merely discussing one or both choices later in the reply.
    first = re.split(r"[.!?\n;]", text.strip(), maxsplit=1)[0]
    first = norm(first)
    for q, word in options:
        if first in (word, "the answer is " + word, "the correct answer is " + word):
            return q
    return None


def attested(text, result_rows, returned=None):
    content = " " + norm(text) + " "
    observed = (
        [{norm(str(x)) for x in r if x is not None} for r in returned]
        if returned is not None
        else None
    )
    for source in result_rows:
        vals = {norm(str(x)) for x in source if x is not None}
        vals.difference_update({"", "yes", "no", "true", "false"})
        mentioned = {v for v in vals if " " + v + " " in content}
        if len(mentioned) >= 2 and (
            observed is None or any(len(o & mentioned) >= 2 for o in observed)
        ):
            return True
    return False
