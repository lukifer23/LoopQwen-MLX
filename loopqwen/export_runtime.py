"""Loader for a self-contained merged text export, also bundled in its runtime/."""

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.tokenizer_utils import load as load_tokenizer

from .looped_model import LoopedModel
from .recurrence import Span


def load_export(directory):
    from .worker import acquire

    acquire()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("The exported model requires native MLX Metal")
    mx.set_cache_limit(256 * 1024**2)
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg.get("export_schema", 1) not in (1, 2)
        or cfg["model_type"] != "qwen3_5_looped"
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
    if "attention_reader_metadata_sha256" in cfg:
        import hashlib

        from .attention_memory import load_reader

        if (
            hashlib.sha256(
                (directory / "attention-reader.json").read_bytes()
            ).hexdigest()
            != cfg["attention_reader_metadata_sha256"]
        ):
            raise ValueError("Exported attention reader metadata differs")
        load_reader(model, directory)
    eos = cfg["text_config"]["eos_token_id"]
    return model, load_tokenizer(directory, eos_token_ids=[eos])


def verify(directory):
    import hashlib

    import numpy as np

    root = Path(directory)
    audit = json.loads((root / "verification.json").read_text())
    model, tokenizer = load_export(root)
    ids = tokenizer.encode(audit["prompt"])
    if [ids] != audit["input_ids"]:
        raise ValueError("Export tokenizer mismatch")
    logits = model(mx.array([ids])).astype(mx.float32)
    actual = hashlib.sha256(np.asarray(logits).tobytes()).hexdigest()
    if actual != audit["logits_sha256"]:
        raise ValueError("Fresh-process export logits differ from merged logits")
    if "reader_probe" in audit:
        from .evaluation.generation_eval import generate

        probe = audit["reader_probe"]
        sample = generate(model, tokenizer, probe["prompt"], 4)
        if (
            sample.get("attention_memory") != probe["section"]
            or sample["token_ids"] != probe["token_ids"]
        ):
            raise ValueError("Standalone full-history attention generation differs")
