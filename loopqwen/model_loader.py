from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load

from . import MODEL, REVISION
from .io import read


def download():
    patterns = ["*.safetensors", "*.json", "*.txt", "LICENSE"]
    return Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=patterns))


def local_snapshot():
    from huggingface_hub import try_to_load_from_cache

    file = try_to_load_from_cache(
        MODEL, "model.safetensors-00001-of-00001.safetensors", revision=REVISION
    )
    return Path(file).parent if isinstance(file, str) else download()


def count(model, trainable=False):
    pairs = tree_flatten(
        model.trainable_parameters() if trainable else model.parameters()
    )
    return sum(v.size for v in {id(a): a for _, a in pairs}.values())


def load_base():
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
