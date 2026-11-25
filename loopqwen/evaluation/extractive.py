"""Completed first-turn extractive QA using the SQuAD v1.1 normalization."""

import re
import string
from collections import Counter


def normalized(s):
    lowered = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", lowered).split())


def score(prediction, aliases):
    if not prediction or not aliases:
        return False, 0.0
    guess = normalized(prediction)
    targets = [normalized(a) for a in aliases]
    words = Counter(guess.split())
    best = 0.0
    for target in targets:
        ref = Counter(target.split())
        common = (words & ref).total()
        if common:
            best = max(best, 2 * common / (words.total() + ref.total()))
    return guess in targets, best


def answer(ids, text):
    if (
        not ids
        or ids[-1] not in (248044, 248046)
        or 248068 in ids
        or ids.count(248069) != 1
    ):
        return None
    if {248044, 248046}.intersection(ids[:-1]):
        raise ValueError("Tokens after assistant boundary")
    boundary = "<|im_end|>" if ids[-1] == 248046 else "<|endoftext|>"
    if not text.endswith(boundary) or text.count("</think>") != 1:
        raise ValueError("Decoded thinking boundary differs")
    visible = text.removesuffix(boundary).split("</think>")[1].strip()
    return (
        re.sub(r"^(?:final\s+)?answer\s*:\s*", "", visible, count=1, flags=re.I).strip()
        or None
    )
