import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .config import span
from .io import checkpoint_location as location
from .io import read, save
from .kernels.gdn_backend import select
from .looped_model import LoopedModel
from .model_loader import load_base
from .training.adapters import attach


def setup(c):
    base, tok, info = load_base()
    model = LoopedModel(base, span(c), c["alpha_init"])
    mx.random.seed(c["seed"])
    attach(model, c["target"], c["rank"], c["lora_alpha"], c["projections"])
    select(model, c["gdn_backend"])
    model.checkpointing = c["gradient_checkpointing"]
    return model, tok, info


def write(root, model, optimizer, state):
    root = Path(root)
    tmp = Path(tempfile.mkdtemp(prefix=".saving-", dir=root))
    mx.save_safetensors(
        str(tmp / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    mx.save_safetensors(
        str(tmp / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    save(tmp / "state.json", state)
    name = f"step-{state['step']:06d}-{tmp.name[-8:]}"
    tmp.rename(root / name)
    save(root / "latest.json", {"directory": name})


def restore(root, model, optimizer=None):
    path = location(root)
    weights = mx.load(str(path / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if weights.keys() != expected.keys() or any(
        v.shape != expected[k].shape for k, v in weights.items()
    ):
        raise ValueError("Checkpoint trainable tensor set mismatch")
    model.load_weights(list(weights.items()), strict=False)
    if optimizer is not None:
        optimizer.state = tree_unflatten(
            list(mx.load(str(path / "optimizer.safetensors")).items())
        )
    return read(path / "state.json")


def load_trained(root):
    state = read(location(root) / "state.json")
    model, tok, _ = setup(state["config"])
    restore(root, model)
    model.alpha_override = state.get("alpha_override")
    model.eval()
    return model, tok, state
