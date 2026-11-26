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
        cfg.get("export_schema", 1) not in (1, 2, 3, 4, 5)
        or cfg["model_type"] != "qwen3_5_looped"
        or cfg["recurrent_cache_version"] != 1
        or cfg["weight_sharing"] is not True
    ):
        raise ValueError("Unsupported looped export schema")
    base = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
    model = LoopedModel(base, Span(**cfg["span"]))
    eos = cfg["text_config"]["eos_token_id"]
    tokenizer = load_tokenizer(directory, eos_token_ids=[eos])
    if cfg.get("export_schema") in (4, 5):
        import hashlib

        from mlx.utils import tree_flatten

        from .attention_memory import load_reader
        from .context_policy import attach as attach_context
        from .training.adapters import attach

        label = "context-policy" if cfg["export_schema"] == 4 else "deliberation"
        metadata = directory / (label + ".json")
        expected = (
            cfg["context_policy_sha256"]
            if cfg["export_schema"] == 4
            else cfg["deliberation_sha256"]
        )
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != expected:
            raise ValueError("Exported policy metadata differs")
        base.load_weights(str(directory / "model.safetensors"), strict=True)
        c = cfg["adapter_config"]
        attach(model, c["target"], c["rank"], c["lora_alpha"], c["projections"])
        weights = mx.load(str(directory / "adapters.safetensors"))
        if set(weights) != set(dict(tree_flatten(model.trainable_parameters()))):
            raise ValueError("Exported register adapter tensor set differs")
        model.load_weights(list(weights.items()), strict=False)
        model.alpha_override = cfg["alpha_override"]
        if cfg["export_schema"] == 4:
            load_reader(model, directory)
            attach_context(model, directory, tokenizer)
        else:
            from .deliberation import attach as attach_deliberation

            spec = json.loads(metadata.read_bytes())
            with (directory / "adapters.safetensors").open("rb") as f:
                if (
                    hashlib.file_digest(f, "sha256").hexdigest()
                    != spec["adapter_sha256"]
                ):
                    raise ValueError("Learned exit adapter differs")
            attach_deliberation(model, spec, directory=directory)
    else:
        model.load_weights(str(directory / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.alpha_override = cfg["alpha_override"]
    model.eval()
    if cfg.get("export_schema") == 3:
        import hashlib

        from .deliberation import attach

        original = directory / "ordinary-projections.safetensors"
        with original.open("rb") as handle:
            if (
                hashlib.file_digest(handle, "sha256").hexdigest()
                != cfg["ordinary_projections_sha256"]
            ):
                raise ValueError("Exported ordinary-depth projections differ")
        attach(
            model,
            cfg["deliberation"],
            original_weights=mx.load(str(original)),
            directory=directory,
        )
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
    return model, tokenizer


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
    if "bank_logits" in audit:
        views = model.context_policy.views()
        if set(views) | {"register"} != set(audit["bank_logits"]):
            raise ValueError("Exported policy bank inventory changed")
        for name, bank in views.items():
            actual = bank(mx.array([ids])).astype(mx.float32)
            if (
                hashlib.sha256(np.asarray(actual).tobytes()).hexdigest()
                != audit["bank_logits"][name]
            ):
                raise ValueError("Exported context decoder logits differ")
    if "ordinary_logits_sha256" in audit:
        ordinary = model.ordinary_depth(mx.array([ids])).astype(mx.float32)
        if (
            hashlib.sha256(np.asarray(ordinary).tobytes()).hexdigest()
            != audit["ordinary_logits_sha256"]
        ):
            raise ValueError(
                "Exported ordinary-depth logits differ from the pinned control"
            )
    for name, route in (
        ("evidence_probe", "deduction"),
        ("induction_probe", "native_induction"),
    ):
        if name not in audit:
            continue
        from .evaluation.generation_eval import generate

        probe = audit[name]
        val = generate(model, tokenizer, probe["prompt"], 32, stop_ids=(248046,))
        if val["context_route"] != route or val["token_ids"] != probe["tokens"]:
            raise ValueError("Exported evidence-loop generation differs")
    if "reader_probe" in audit:
        from .evaluation.generation_eval import generate

        probe = audit["reader_probe"]
        sample = generate(model, tokenizer, probe["prompt"], 4)
        if (
            sample.get("attention_memory") != probe["section"]
            or sample["token_ids"] != probe["token_ids"]
        ):
            raise ValueError("Standalone full-history attention generation differs")
