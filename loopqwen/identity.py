"""Content identities for controlled training and evaluation."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from .io import digest

PACKAGE = Path(__file__).resolve().parent
TRAIN_FILES = (
    "__init__.py",
    "config.py",
    "model_loader.py",
    "looped_model.py",
    "recurrence.py",
    "kernels/gdn_backend.py",
    "training/trainer.py",
    "training/adapters.py",
    "training/optimizers.py",
    "training/losses.py",
    "data/sampling.py",
    "data/tokenize.py",
    "data/v2.py",
    "checkpoint.py",
    "identity.py",
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dependencies():
    return {
        name: version(name)
        for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy")
    }


def training_identity(config, manifest):
    return dict(
        schema=1,
        sources={p: digest(PACKAGE / p) for p in TRAIN_FILES},
        dependencies=dependencies(),
        config=fingerprint(config),
        data=fingerprint(manifest),
        optimizer="split-adamw-v1" if "gate_lr" in config else "adamw-v1",
        sampler="v2-ordered-batches-v1",
        recurrence="shared-site-v1",
    )


def evaluation_identity(manifest, generation, items, decoding, scorer):
    state = json.loads((generation / "state.json").read_text())
    sources = {
        str(p.relative_to(PACKAGE)): digest(p) for p in sorted(PACKAGE.rglob("*.py"))
    }
    return dict(
        schema=3,
        experiment=fingerprint(manifest),
        weights=digest(generation / "adapters.safetensors"),
        config=fingerprint(state["config"]),
        metadata=digest(generation / "state.json"),
        sources=fingerprint(sources),
        scorer=scorer,
        data=fingerprint(state["manifest"]),
        items=fingerprint(items),
        tokenizer=state["config"]["tokenizer_revision"],
        decoding=decoding,
        dependencies=dependencies(),
        backend="mlx-metal",
    )
