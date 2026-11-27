"""Build and load a standalone eight-bit copy of the trained six-layer recurrent reader."""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.tuner.lora import LoRALinear

from . import REVISION
from .checkpoint import restore
from .config import span
from .export_runtime import load_export_tokenizer
from .hardware import require_metal
from .io import digest, read, save
from .looped_model import LoopedModel
from .ordinary_model import ordinary
from .recurrence import Span
from .training.adapters import attach
from .worker import acquire

ADAPTER = "1d69dd71119f57503321dd361348acee5bb9989ac92e82a7f9309a06e9bc4c68"
STATE = "55c1610690b2bd8088d001811f1c8ce4a9e6c3daabd8cef387dfaea5c4920559"
MARKER = "compact-entailment.json"


def quantize(text):
    nn.quantize(
        text,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, (nn.Linear, nn.Embedding)),
    )


def load(directory):
    require_metal()
    acquire()
    directory = Path(directory)
    cert = read(directory / MARKER)
    if cert["adapter_sha256"] != ADAPTER or cert["base_revision"] != REVISION:
        raise ValueError("Different compact reader lineage")
    for name, checksum in cert["files"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Compact reader payload differs at {name}")
    cfg = read(directory / "config.json")
    text = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
    quantize(text)
    model = LoopedModel(text, Span(**cfg["span"]))
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tok = load_export_tokenizer(
        directory, eos_token_ids=[cfg["text_config"]["eos_token_id"]]
    )
    return model, tok


def build(release, adapter, directory):
    if directory.exists():
        raise FileExistsError(directory)
    if (
        digest(adapter / "adapters.safetensors") != ADAPTER
        or digest(adapter / "state.json") != STATE
    ):
        raise ValueError(
            "The compact reader requires the qualified entailment checkpoint"
        )
    require_metal()
    acquire()
    start = perf_counter()
    mx.reset_peak_memory()
    base, tok, _ = ordinary(release)
    if base.base_source != "certified_release_text":
        raise ValueError("Use the certified release 05 weights")
    c = read(adapter / "state.json")["spec"]["deployment"]["config"]
    model = LoopedModel(base.text, span(c), c["alpha_init"])
    attach(model, c["target"], c["rank"], c["lora_alpha"], c["projections"])
    restore(adapter, model)
    model.eval()
    ids = mx.array(
        [tok.encode("The meeting took place on Tuesday.", add_special_tokens=False)]
    )

    def logits():
        return model.project(model.hidden(ids)[:, -1]).astype(mx.float32)

    original = logits()
    mx.eval(original)
    fused = []
    for i, layer in enumerate(model.text.layers):
        for name, module in list(layer.named_modules()):
            if isinstance(module, LoRALinear):
                parent, key = name.rsplit(".", 1)
                setattr(getattr(layer, parent), key, module.fuse())
                fused.append(f"{i}.{name}")
    quantize(model.text)
    after = logits()
    rms = mx.sqrt(
        mx.mean((after - original).square()) / mx.mean(original.square())
    ).item()
    if rms >= 0.06 or not 0 < model.effective_alpha().item() < 1 or not fused:
        raise ValueError(
            f"Compact reader failed the independent conversion check ({rms=})"
        )
    directory.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=directory.parent) as tmp:
        root = Path(tmp) / "model"
        root.mkdir()
        model.save_weights(str(root / "model.safetensors"))
        cfg = read(release / "config.json")
        save(
            root / "config.json",
            dict(
                model_type="qwen3_5_looped",
                text_config=cfg["text_config"],
                span=asdict(model.span),
            ),
        )
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "LICENSE",
            "chat_template.jinja",
            "special_tokens_map.json",
            "added_tokens.json",
        ):
            if (release / name).is_file():
                shutil.copyfile(release / name, root / name)
        if (release / "chat_templates").is_dir():
            shutil.copytree(release / "chat_templates", root / "chat_templates")
        files = {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        cert = dict(
            base_revision=REVISION,
            adapter_sha256=ADAPTER,
            files=files,
            fused=fused,
            relative_logit_rms=rms,
            learned_gate=model.effective_alpha().item(),
            seconds=perf_counter() - start,
            peak_bytes=mx.get_peak_memory(),
        )
        save(root / MARKER, cert)
        root.rename(directory)
    return cert


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "adapter", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    print(build(args.release, args.adapter, args.output))
