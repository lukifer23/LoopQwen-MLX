"""Eight-node pointer maps with exact step labels and disjoint map identities."""

import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from ..io import read, save, digest

STATES = 8
ROOT = Path("data/v4_pointer")


def states(mapping, start, count):
    x = start
    result = []
    for _ in range(count):
        x = mapping[x]
        result.append(x)
    # Independent one-hot transition-matrix multiplication.
    vector = [int(i == start) for i in range(STATES)]
    for gold in result:
        vector = [
            sum(vector[i] for i in range(STATES) if mapping[i] == j)
            for j in range(STATES)
        ]
        if vector.index(1) != gold:
            raise ArithmeticError("Independent state verification failed")
    return result


def render(row, n, verbal=False):
    if verbal:
        links = ". ".join(
            f"Room {i} leads to room {j}" for i, j in enumerate(row["map"])
        )
        return (
            f"{links}. Starting in room {row['start']}, follow {n} links. Destination:"
        )
    links = " ".join(f"{i}>{j}" for i, j in enumerate(row["map"]))
    return f"Map: {links}\nStart: {row['start']}  Hops: {n}\nDestination:"


def build():
    dest = ROOT
    if dest.exists():
        meta = read(dest / "manifest.json")
        for split, h in meta["splits"].items():
            if digest(dest / (split + ".jsonl")) != h:
                raise ValueError("Changed pointer split")
        return meta
    dest.parent.mkdir(exist_ok=True)
    seen = set()
    with TemporaryDirectory(dir=dest.parent) as temp:
        tmp = Path(temp) / "dataset"
        tmp.mkdir()
        meta = dict(
            schema=1,
            seed=40819,
            states=STATES,
            license="Project MIT",
            generator=digest(Path(__file__)),
            splits={},
        )
        for split, count in (("train", 3840), ("development", 384)):
            rng = random.Random(f"40819/{split}")
            with (tmp / (split + ".jsonl")).open("w") as f:
                for _ in range(count):
                    for attempt in range(500):
                        mapping = rng.sample(range(STATES), STATES)
                        key = tuple(mapping)
                        start = rng.randrange(STATES)
                        gold = states(mapping, start, 6)
                        if key not in seen and start not in gold[:3]:
                            break
                    else:
                        raise RuntimeError("Pointer generator exhausted")
                    seen.add(key)
                    row = dict(
                        id=hashlib.sha256(bytes(mapping)).hexdigest(),
                        map=mapping,
                        start=start,
                        states=gold,
                    )
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            meta["splits"][split] = digest(tmp / (split + ".jsonl"))
        save(tmp / "manifest.json", meta)
        tmp.rename(dest)
    return meta
