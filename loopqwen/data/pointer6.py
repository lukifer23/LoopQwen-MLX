"""Phase 6 immutable rendering protocol over the original pointer splits."""

import random
from pathlib import Path
from . import pointer5
from ..identity import fingerprint
from ..io import digest, read, save

SPEC = read("experiments/phase6.json")
ROOT = Path("data") / SPEC["dataset"]


def ordered(row, policy, seed, epoch=0):
    order = list(range(len(row["map"])))
    if policy == "reverse":
        order.reverse()
    elif policy.startswith("rotate"):
        shift = int(policy[6:]) % len(order)
        order = order[shift:] + order[:shift]
    elif policy != "canonical":
        if policy not in {"randomized", "mixed", *SPEC["orders"]}:
            raise ValueError(policy)
        rng = random.Random(
            fingerprint([seed, epoch, row["map_id"], row["example_id"], policy])
        )
        if policy != "mixed" or rng.random() >= SPEC["mixed_canonical_fraction"]:
            rng.shuffle(order)
    return {**row, "order": order}


def build():
    parent = pointer5.build()
    published = read("docs/measurements/phase5/dataset.json")
    if parent != published:
        raise ValueError("Phase 5 map streams differ from publication")
    value = dict(
        schema=1,
        parent=parent["files"],
        renderer=digest(Path(__file__)),
        registration=fingerprint(SPEC),
        evaluation_seed=SPEC["evaluation_seed"],
        orders=SPEC["orders"],
        train_maps=3840,
        development_maps=384,
    )
    path = ROOT / "manifest.json"
    if path.exists() and read(path) != value:
        raise ValueError("Incompatible rendering namespace")
    if not path.exists():
        save(path, value)
    return value


def epoch_batches(train, seed, epoch):
    order = list(range(len(train)))
    random.Random(seed + epoch).shuffle(order)
    batch = SPEC["batch"]
    return [
        [train[i] for i in order[j : j + batch]] for j in range(0, len(order), batch)
    ]


def depth(step, epoch):
    return 1 + (step + epoch) % (2 if step < 48 else 3)
