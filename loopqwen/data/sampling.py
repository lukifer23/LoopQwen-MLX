from collections import Counter
import hashlib
import json
import random

from . import v2
from .build_data import digest
from ..io import read

MIXTURES = {
    "answer_only": {"answer": 1},
    "verified_traces": {"trace": 1},
    "mixed_answer_and_traces": {"answer": 0.5, "trace": 0.5},
    "traces_with_general_language_retention": {"trace": 0.9, "language": 0.1},
}


def prepare(c):
    rng = random.Random(c["seed"])
    mix = c.get("mixture_weights", MIXTURES[c["mixture"]])
    if (
        set(mix) - {"answer", "trace", "language", "gsm"}
        or min(mix.values()) < 0
        or abs(sum(mix.values()) - 1) > 1e-7
    ):
        raise ValueError(
            "Mixture weights must be nonnegative probabilities summing to one"
        )
    math_weight = mix.get("answer", 0) + mix.get("trace", 0)
    if math_weight <= 0:
        raise ValueError("Include procedural examples")
    manifest = read(v2.ROOT / "manifest.json")
    path = v2.ROOT / "train.jsonl"
    if digest(path) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training split changed")
    pool = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        variants = r.pop("variants")
        mode = (
            "answer"
            if int(r["id"][:12], 16) / 16**12 < mix.get("answer", 0) / math_weight
            else "trace"
        )
        pool.append(
            r
            | variants[mode]
            | dict(protocol=mode, bucket_length=variants["trace"]["input_length"])
        )
    count = len(pool)
    for source in ("language", "gsm"):
        if not mix.get(source):
            continue
        if source == "language":
            extra = v2.rows("language_train", "language")
        else:
            from .public_traces import gsm

            extra, manifest["gsm_train"] = gsm(c["sequence_length"])
        rng.shuffle(extra)
        if not extra:
            raise ValueError("Empty mixture source " + source)
        pool.extend(
            extra[i % len(extra)]
            for i in range(round(count * mix[source] / math_weight))
        )
    excluded = Counter(
        r["category"] for r in pool if r["input_length"] > c["sequence_length"]
    )
    pool = [r for r in pool if r["input_length"] <= c["sequence_length"]]
    if not pool:
        raise ValueError("No eligible samples")
    rng.shuffle(pool)
    if c.get("bucket", True):
        size = c["batch_size"]
        width = size * 16
        groups = []
        for start in range(0, len(pool), width):
            window = sorted(
                pool[start : start + width],
                key=lambda r: r.get("bucket_length", r["input_length"]),
            )
            groups.extend(window[j : j + size] for j in range(0, len(window), size))
        rng.shuffle(groups)
        pool = [r for group in groups for r in group]
    policy = dict(
        eligible=len(pool),
        excluded=dict(excluded),
        weights=mix,
        schedule_digest=hashlib.sha256(
            "".join(r["id"] + r["protocol"] for r in pool).encode()
        ).hexdigest(),
    )
    valid = v2.rows("validation", "trace" if mix.get("trace", 0) else "answer")[
        : c["eval_count"]
    ]
    return pool, valid, manifest, policy
