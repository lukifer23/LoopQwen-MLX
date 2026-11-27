from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load

from . import MODEL, REVISION
from .hardware import require_metal
from .io import read


def download():
    patterns = ["*.safetensors", "*.json", "*.txt", "LICENSE"]
    return Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=patterns))


def local_snapshot():
    from huggingface_hub import try_to_load_from_cache

    file = try_to_load_from_cache(
        MODEL, "model.safetensors-00001-of-00001.safetensors", revision=REVISION
    )
    if isinstance(file, str):
        path = Path(file).parent
        if all(
            (path / name).is_file()
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            )
        ):
            return path
    return download()


def count(model, trainable=False):
    pairs = tree_flatten(
        model.trainable_parameters() if trainable else model.parameters()
    )
    return sum(v.size for v in {id(a): a for _, a in pairs}.values())


def load_base():
    require_metal()
    from .worker import acquire

    acquire()
    path = local_snapshot()
    start = perf_counter()
    outer, tokenizer = load(str(path))
    model = outer.language_model
    model.eval()
    config = read(path / "config.json")["text_config"]
    types = [
        "linear_attention" if layer.is_linear else "full_attention"
        for layer in model.layers
    ]
    if (
        types != config["layer_types"]
        or len(types) != 24
        or model.args.vocab_size != 248320
    ):
        raise ValueError(
            "Loaded decoder differs from the official 0.8B text configuration"
        )
    params = tree_flatten(model.parameters())
    dtype_counts = Counter()
    for _, arr in params:
        dtype_counts[str(arr.dtype)] += arr.size
    info = dict(
        model=MODEL,
        revision=REVISION,
        unique_parameters=count(model),
        precision=dtype_counts,
        load_seconds=perf_counter() - start,
        layers=[
            dict(
                index=i,
                kind=kind,
                projections=[
                    n for n, m in layer.named_modules() if isinstance(m, nn.Linear)
                ],
            )
            for i, (layer, kind) in enumerate(zip(model.layers, types))
        ],
    )
    return model, tokenizer, info


def ordinary(directory=None):
    """Load pinned ordinary-depth weights from a certified text export or its original snapshot."""
    from .looped_model import LoopedModel
    from .recurrence import Span

    inventory = Path(directory) / "tool-export-files.json" if directory else None
    if inventory is not None and inventory.is_file():
        from .io import digest

        if (
            digest(inventory)
            == "e296d289a70a1dbe4de592fda55d6c650e7de46fcb2b38aa110b6199d7d15a3f"
        ):
            return _ordinary_from_release(inventory.parent)
    text, tokenizer, _ = load_base()
    model = LoopedModel(text, Span(passes=1))
    object.__setattr__(model, "base_source", "official_snapshot")
    return model, tokenizer, REVISION


def _ordinary_from_release(directory):
    """All 320 exported text tensors were compared exactly with the official native load."""
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from .export_runtime import load_export_tokenizer
    from .io import digest
    from .looped_model import LoopedModel
    from .recurrence import Span
    from .worker import acquire

    require_metal()
    acquire()
    files = read(directory / "tool-export-files.json")
    members = {
        "model.safetensors",
        "provenance.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    }
    members.update(k for k in files if k.startswith("chat_templates/"))
    members.update(
        str(p.relative_to(directory))
        for p in (directory / "chat_templates").glob("*.jinja")
    )
    for name in members:
        path = directory / name
        if path.exists() != (name in files) or (
            name in files and digest(path) != files[name]
        ):
            raise ValueError("The certified Base file inventory differs at " + name)
    if read(directory / "provenance.json")["base_revision"] != REVISION:
        raise ValueError("Wrong original Base revision")
    config = read(directory / "config.json")["text_config"]
    text = TextModel(TextModelArgs.from_dict(config))
    text.load_weights(str(directory / "model.safetensors"), strict=True)
    mx.eval(text.parameters())
    text.eval()
    model = LoopedModel(text, Span(passes=1))
    object.__setattr__(model, "base_source", "certified_release_text")
    tokenizer = load_export_tokenizer(directory, eos_token_ids=[config["eos_token_id"]])
    return model, tokenizer, REVISION
