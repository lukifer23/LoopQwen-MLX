"""Loader for a self-contained merged text export, also bundled in its runtime/."""

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.tokenizer_utils import load as load_tokenizer

from .looped_model import LoopedModel
from .recurrence import Span


def load_export(directory):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("The exported model requires native MLX Metal")
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg["model_type"] != "qwen3_5_looped"
        or cfg["recurrent_cache_version"] != 1
        or cfg["weight_sharing"] is not True
    ):
        raise ValueError("Unsupported looped export schema")
    base = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
    model = LoopedModel(base, Span(**cfg["span"]))
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.alpha_override = cfg["alpha_override"]
    model.eval()
    eos = cfg["text_config"]["eos_token_id"]
    return model, load_tokenizer(directory, eos_token_ids=[eos])
