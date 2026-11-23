import re
from pathlib import Path

import yaml

from . import MODEL, REVISION
from .recurrence import Span


def span(c):
    return Span(**{k: c[k] for k in ("start", "length", "passes", "mode")})


def config(path):
    root = Path("configs")
    defaults = yaml.safe_load((root / "defaults.yaml").read_text())
    user = yaml.safe_load(Path(path).read_text())
    if (
        extra := user.keys()
        - defaults.keys()
        - {
            "preset",
            "name",
            "packing",
            "mixture",
            "mixture_weights",
            "bucket",
            "schedule",
            "warmup_fraction",
            "min_lr",
            "ramp_fraction",
            "label_budget",
            "seconds_budget",
            "validation_batch_size",
        }
    ):
        raise ValueError(f"Unknown config keys {extra}")
    presets = yaml.safe_load((root / "presets.yaml").read_text())
    c = defaults | presets[user.get("preset", "mac_safe")] | user
    span(c).sites(24)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", c["name"]):
        raise ValueError("Invalid run name")
    if (
        c["base_model"],
        c["base_revision"],
        c["tokenizer_revision"],
        c["precision"],
        c["optimizer"],
        c["training_split"],
        c["validation_split"],
    ) != (
        MODEL,
        REVISION,
        REVISION,
        "bfloat16",
        "AdamW",
        "train",
        "validation",
    ):
        raise ValueError(
            "This experiment requires the pinned BF16 base, tokenizer, AdamW and registered train/validation splits."
        )
    for key in (
        "steps",
        "rank",
        "batch_size",
        "accumulation",
        "checkpoint_interval",
        "eval_interval",
        "eval_count",
        "memory_target_gib",
        "min_available_gib",
        "learning_rate",
        "clip",
    ):
        if c[key] <= 0:
            raise ValueError(key)
    if (
        c["sequence_length"] < 16
        or c["warmup"] < 0
        or c["gdn_backend"] not in ("reference", "native")
    ):
        raise ValueError("Invalid sequence length, warmup or GDN backend")
    if (
        c["curriculum"] not in ("direct", "gradual", "mixed")
        or len(c["depth_weights"]) != 3
        or min(c["depth_weights"]) < 0
        or sum(c["depth_weights"]) <= 0
    ):
        raise ValueError("Invalid curriculum")
    if c["alpha_schedule"] is not None and (
        len(c["alpha_schedule"]) != 2
        or not all(0 <= a <= 1 for a in c["alpha_schedule"])
        or c["mode"] != "stabilized"
    ):
        raise ValueError(
            "alpha_schedule needs a stabilized loop and two endpoints in [0,1]"
        )
    if c["dataset"] not in ("v2", "pilot"):
        raise ValueError("Unknown dataset")
    if c["dataset"] == "v2":
        from .data.sampling import MIXTURES

        if not c["response_only"]:
            raise ValueError(
                "V2 source accounting requires its registered target masks"
            )
        if c.get("mixture") not in MIXTURES:
            raise ValueError("Choose a V2 mixture")
    if (
        c.get("schedule", "constant") not in ("cosine", "constant")
        or not 0 <= c.get("min_lr", 0) <= c["learning_rate"]
    ):
        raise ValueError("Invalid LR schedule")
    for key in ("warmup_fraction", "ramp_fraction"):
        if key in c and not 0 < c[key] <= 1:
            raise ValueError(key)
    if c.get("validation_batch_size", 1) not in (1, 2, 4) or (
        "validation_batch_size" in c and c["dataset"] != "v2"
    ):
        raise ValueError("V2 validation batch size must be 1, 2 or 4")
    for key in ("label_budget", "seconds_budget"):
        if key in c and c[key] <= 0:
            raise ValueError(key)
    return c
