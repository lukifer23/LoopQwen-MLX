"""Matched native Qwen middle-span and ordinary-depth decoders."""
from pathlib import Path

import mlx.core as mx
from huggingface_hub import try_to_load_from_cache
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

from .export_runtime import load_export_tokenizer
from .hardware import require_metal
from .io import digest, read
from .kernels.gdn_backend import select
from .looped_model import LoopedModel
from .ordinary_model import ordinary
from .recurrence import Span
from .tool_model import Assistant
from .worker import acquire

INSTRUCTION_REV = "2fc06364715b967f1860aea9cf38778875588b17"
INSTRUCTION_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
SHARD = "model.safetensors-00001-of-00001.safetensors"


def donor(directory, which):
    require_metal()
    acquire()
    if which == "base":
        model, tokenizer, _ = ordinary(directory, require_release=True)
        return model.text, tokenizer
    if which != "instruction":
        raise ValueError("Unrecognized donor")
    file = try_to_load_from_cache("Qwen/Qwen3.5-0.8B", SHARD, revision=INSTRUCTION_REV)
    if not isinstance(file, str) or digest(Path(file)) != INSTRUCTION_SHA:
        raise ValueError("Install the pinned official instruction snapshot")
    cfg = read(Path(directory) / "config.json")["text_config"]
    if read(Path(file).parent / "config.json")["text_config"] != cfg:
        raise ValueError("Incompatible donor text configuration")
    tokenizer = load_export_tokenizer(directory, eos_token_ids=[cfg["eos_token_id"]])
    outer, _ = load(str(Path(file).parent))
    outer.eval()
    return outer.language_model, tokenizer


def architecture(text, depth, seed):
    """Zero-output LoRA leaves the ordinary exit identical at initialization."""
    mx.random.seed(seed)
    model = LoopedModel(text, Span(12, 4, depth, "input_anchor"), alpha=0.8)
    model.freeze()
    if depth > 1:
        model.unfreeze(keys=["alpha_logit"], recurse=False)
    for i in (12, 13, 14, 15, 19, 23):
        mlp = model.text.layers[i].mlp
        mlp.down_proj = LoRALinear.from_base(mlp.down_proj, r=16, scale=2, dropout=0)
    select(model, "native")
    model.checkpointing = True
    model.eval()
    return model


class MiddleAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        path = Path(policy)
        spec = read(path / "registration.json")
        if digest(path / "adapters.safetensors") != spec["weights"]:
            raise ValueError("Middle-span adapter weights differ")
        text, self.tokenizer = donor(directory, spec["donor"])
        model = architecture(text, spec["trained_depth"], spec["seed"])
        weights = mx.load(str(path / "adapters.safetensors"))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if expected.keys() != weights.keys() or any(
            v.shape != expected[k].shape or v.dtype != expected[k].dtype
            for k, v in weights.items()
        ):
            raise ValueError("Incompatible middle-span adapter")
        model.load_weights(list(weights.items()), strict=False)
        model.span = Span(12, 4, spec["serving_depth"], "input_anchor")
        model.eval()
        self._tool_model = model
        self.reuse_prefix = True
