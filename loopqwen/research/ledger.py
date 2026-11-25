"""A complete, uniformly keyed ledger with a token-length floor and fresh values."""

import random

from ..evaluation.language7 import prompt_ids

WORDS = [
    "amber",
    "ash",
    "blue",
    "brass",
    "brook",
    "cedar",
    "chalk",
    "clay",
    "cloud",
    "coral",
    "crane",
    "dawn",
    "deer",
    "elm",
    "fern",
    "flax",
    "frost",
    "gold",
    "gray",
    "green",
    "hill",
    "iris",
    "jade",
    "lake",
    "lime",
    "maple",
    "mint",
    "oak",
    "pine",
    "plum",
    "reed",
    "rose",
    "sand",
    "silk",
    "slate",
    "snow",
    "stone",
    "swan",
    "teal",
    "thorn",
    "vine",
    "wave",
    "willow",
    "wolf",
    "wren",
    "zinc",
]


def ledger(t, seed, length, fraction, announce):
    rng = random.Random(seed)
    # Keys and values are independently permuted. Their ordinals carry no label information.
    count = min(length // 6 + 12, 900000, len(WORDS) ** 3)
    names = rng.sample(range(len(WORDS) ** 3), count)
    numbers = rng.sample(range(100000, 1000000), count)

    def record(i):
        n = names[i]
        key = " ".join(WORDS[(n // len(WORDS) ** j) % len(WORDS)] for j in (2, 1, 0))
        return key, str(numbers[i])

    key, gold = record(0)
    head = f"Retrieve the value for {key}.\n" if announce else ""
    head += "Ledger entries (key = value):\n"
    tail = f"End of ledger.\nWhat value is registered for {key}?"
    overhead = len(prompt_ids(t, {"question": head + tail}, {"demos": 0}))
    lines = []
    total = overhead
    for i in range(1, count):
        name, value = record(i)
        line = f"{name} = {value}\n"
        lines.append(line)
        total += len(t.encode(line, add_special_tokens=False))
        if total >= length:
            break
    index = min(len(lines) - 1, int(len(lines) * fraction))
    target = f"{key} = {gold}\n"
    lines[index] = target
    while True:
        question = head + "".join(lines) + tail
        ids = prompt_ids(t, {"question": question}, {"demos": 0})
        if len(ids) >= length:
            break
        name, value = record(len(lines) + 1)
        lines.append(f"{name} = {value}\n")
    if (
        len(ids) > length + 32
        or question.count(target) != 1
        or question.count(gold) != 1
    ):
        raise ValueError("Ledger length or unique value differs")
    return {
        "seed": seed,
        "minimum_length": length,
        "fraction": fraction,
        "announce": announce,
        "key": key,
        "gold": gold,
        "prompt_ids": ids,
        "entries": len(lines),
        "index": index,
    }
