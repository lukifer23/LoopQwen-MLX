"""Fresh token-level memory supervision with interchangeable formats and hard distractors."""

import random

from ..evaluation.language7 import prompt_ids
from .ledger import WORDS


def make_item(t, seed, length, fraction, style):
    rng = random.Random(seed)
    count = length // 6 + 80
    radix = len(WORDS)
    choices = rng.sample(range(radix**3), count)
    values = rng.sample(range(100000, 1000000), count)
    names = [
        " ".join(WORDS[(v // radix**j) % radix] for j in (2, 1, 0)) for v in choices
    ]
    key = names[0]
    gold = str(values[0])
    templates = {
        "table": "{} = {}\n",
        "statement": "The archive assigns {} the code {}.\n",
        "filing": "The filing code under {} is {}.\n",
        "note": "The reference for {} reads {}.\n",
    }
    template = templates[style]
    head = "Register:\n"
    tail = f"End of register.\nWhich six-digit code belongs to {key}?"
    total = len(prompt_ids(t, {"question": head + tail}, {"demos": 0}))
    records = []
    for name, v in zip(names[1:], values[1:], strict=True):
        records.append((name, str(v)))
        total += len(t.encode(template.format(name, v), add_special_tokens=False))
        if total >= length:
            break
    idx = int(len(records) * fraction)
    records[idx] = (key, gold)
    used = {n for n, _ in records}
    slots = rng.sample([i for i in range(len(records)) if i != idx], 3)
    for j, pos in enumerate(slots):
        changed = key.split()
        for w in rng.sample(WORDS, len(WORDS)):
            changed[j] = w
            cand = " ".join(changed)
            if cand not in used:
                used.add(cand)
                records[pos] = (cand, records[pos][1])
                break
    while True:
        question = head + "".join(template.format(*r) for r in records) + tail
        ids = prompt_ids(t, {"question": question}, {"demos": 0})
        if len(ids) >= length:
            break
        nxt = len(records) + 1
        while names[nxt] in used:
            nxt += 1
        records.append((names[nxt], str(values[nxt])))
        used.add(names[nxt])
    digits = t.encode(gold, add_special_tokens=False)
    at = [
        i
        for i, x in enumerate(ids)
        if x == digits[0] and ids[i : i + len(digits)] == digits
    ]
    if (
        len({x[0] for x in records}) != len(records)
        or len(at) != 1
        or question.count(gold) != 1
        or len(ids) > length + 48
    ):
        raise ValueError("Register labels, lengths or target identity differ")
    return {
        "seed": seed,
        "minimum_length": length,
        "fraction": fraction,
        "style": style,
        "key": key,
        "gold": gold,
        "prompt_ids": ids,
        "gold_start": at[0],
        "gold_stop": at[0] + len(digits),
        "hard_keys": [records[pos][0] for pos in slots],
    }


def dataset(t, s):
    train = []
    lengths = (384, 640, 896)
    styles = ("table", "statement", "note")
    for i in range(s["updates"] * s["logical_batch"]):
        train.append(
            make_item(
                t,
                s["seed"] + i,
                lengths[i % 3],
                (1 + (i * 37) % 74) / 100,
                styles[(i // 3) % 3],
            )
        )
    calibration = [
        make_item(
            t,
            s["seed"] + 1000 + i,
            1024,
            (0.12, 0.42, 0.75)[i % 3],
            styles[(i // 3) % 3],
        )
        for i in range(12)
    ]
    panel = []
    for n in (2048, 16384):
        for f in (0.10, 0.75):
            for style in ("table", "filing"):
                panel.append(make_item(t, s["seed"] + 2000 + len(panel), n, f, style))
    return train, calibration, panel
