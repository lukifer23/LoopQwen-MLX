"""Bind reusable measurements to the execution, scoring and tokenizer implementations."""

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

from ..identity import PACKAGE, evaluation_identity as base_identity
from ..io import digest
from ..model_loader import local_snapshot

FILES = (
    "checkpoint.py",
    "recurrence.py",
    "recurrent_cache.py",
    "looped_model.py",
    "model_loader.py",
    "config.py",
    "kernels/gdn_backend.py",
    "training/adapters.py",
    "training/losses.py",
    "data/tokenize.py",
    "data/v2.py",
    "evaluation/identity.py",
    "evaluation/phase2.py",
    "evaluation/phase3.py",
    "evaluation/answers.py",
    "evaluation/trace_verifier.py",
    "data/graphs.py",
    "data/token_classes.py",
    "evaluation/generation_eval.py",
    "evaluation/statistics.py",
)


@lru_cache
def runtime():
    snap = local_snapshot()
    upstream = (
        "mlx_lm.models.qwen3_5",
        "mlx_lm.models.gated_delta",
        "mlx_lm.models.cache",
        "mlx_lm.tokenizer_utils",
    )
    return dict(
        sources={name: digest(PACKAGE / name) for name in FILES},
        installed={name: digest(Path(find_spec(name).origin)) for name in upstream},
        tokenizer={
            name: digest(snap / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
        },
    )


def evaluation_identity(*args):
    identity = base_identity(*args)
    identity["sources"] = runtime()
    return identity
