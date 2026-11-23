from collections import Counter
from pathlib import Path
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
    root = Path(c.get("dataset_path", "data/" + c["dataset"]))
    manifest = read(root / "manifest.json")
    path = root / "train.jsonl"
    if digest(path) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training split changed")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if c.get("mixture_policy", "example_fraction") == "target_token_fraction":
        if set(mix) - {"answer", "trace"}:
            raise ValueError(
                "Target-token fractions currently support procedural protocols"
            )
        frequencies = {}
        for mode, fraction in mix.items():
            eligible = [
                r["variants"][mode]["output_length"]
                for r in records
                if r["variants"][mode]["input_length"] <= c["sequence_length"]
            ]
            if not eligible and fraction:
                raise ValueError("No eligible examples for requested " + mode)
            if not fraction:
                continue
            frequencies[mode] = fraction * len(eligible) / sum(eligible)
        total = sum(frequencies.values())
        mix = {mode: f / total for mode, f in frequencies.items()}
    pool = []
    for r in records:
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
            extra = v2.rows("language_train", "language", root)
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
    absent = {k for k, v in mix.items() if v > 0} - {r["protocol"] for r in pool}
    if absent:
        raise ValueError(f"No eligible examples for {sorted(absent)}")
    rng.shuffle(pool)
    if c.get("bucket", True):
        size = c["batch_size"]
        width = size * 16
        policy_name = c.get("bucket_policy", "matched_instance_order")
        keys = {
            "matched_instance_order": lambda r: r.get(
                "bucket_length", r["input_length"]
            ),
            "actual_input_length": lambda r: r["input_length"],
            "protocol_and_length": lambda r: (r["protocol"], r["input_length"]),
        }
        groups = []
        for start in range(0, len(pool), width):
            window = sorted(
                pool[start : start + width],
                key=keys[policy_name],
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
    valid = v2.rows("validation", "trace" if mix.get("trace", 0) else "answer", root)[
        : c["eval_count"]
    ]
    return pool, valid, manifest, policy
