"""Use a trained Base-derived two-layer loop for both plain text and read-only tool turns."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from .export_runtime import load_export_tokenizer
from .io import digest, read
from .observation_review import backbone
from .tool_model import Assistant, tool_view


def model_for(directory, spec):
    directory = Path(directory)
    text = backbone(directory, spec)
    mx.random.seed(spec["fit_seed"])
    model = tool_view(text, spec["start_layer"], spec["rank"], 2)
    tokenizer = load_export_tokenizer(
        directory,
        eos_token_ids=[
            read(Path(directory) / "config.json")["text_config"]["eos_token_id"]
        ],
    )
    return model, tokenizer


class FileEvidenceAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        policy = Path(policy)
        spec = read(policy / "registration.json")
        fit = read(policy / "fit.json")
        checks = policy / "checksums.json"
        if checks.is_file():
            inventory = read(checks)
            if any(
                digest(policy / k) != inventory[k]
                for k in ("registration.json", "fit.json", "tool-adapters.safetensors")
            ):
                raise ValueError("File evidence model inventory changed")
        if (
            not spec.get("file_evidence")
            or spec["start_layer"] != 22
            or fit["passes"] != 2
        ):
            raise ValueError("Expected the trained two-layer file evidence reader")
        weights = policy / "tool-adapters.safetensors"
        if digest(weights) != fit["weights"]:
            raise ValueError("File evidence training weights changed")
        model, self.tokenizer = model_for(directory, spec)
        tensors = mx.load(str(weights))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if tensors.keys() != expected.keys() or any(
            v.shape != expected[k].shape for k, v in tensors.items()
        ):
            raise ValueError("Trained adapter shape differs")
        model.load_weights(list(tensors.items()), strict=False)
        for layer in model.text.layers[model.span.start :]:
            for path, m in list(layer.named_modules()):
                if isinstance(m, LoRALinear):
                    parent, key = path.rsplit(".", 1)
                    setattr(getattr(layer, parent), key, m.fuse())
        mx.eval(model.parameters())
        model.eval()
        self._tool_model = model
        self.reuse_prefix = True
