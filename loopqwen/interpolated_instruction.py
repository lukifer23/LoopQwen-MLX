"""Construct a resident recurrent instruction model from the released trained task update."""

from pathlib import Path

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

from .io import digest, read
from .tool_model import Assistant, bind


class InterpolatedAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        policy = Path(policy)
        spec = read(policy / "registration.json")
        if (policy / "checksums.json").exists() and digest(
            policy / "registration.json"
        ) != read(policy / "checksums.json")["registration.json"]:
            raise ValueError("Instruction policy differs from the sealed registration")
        recipe = spec["instruction_interpolation"]
        folder = self.directory / "tool"
        if digest(folder / "tool-model.json") != recipe["metadata_sha256"]:
            raise ValueError("Instruction policy requires its original trained export")
        strength = recipe["trained_fraction"]
        recurrence = recipe["second_pass_fraction"]
        if not (0 < strength <= 1 and 0 < recurrence <= 1):
            raise ValueError(
                "Both trained and recurrent contributions must be positive"
            )
        model = bind(None, folder)
        layer = model.text.layers[model.span.start]
        if model.span.passes != 2 or model.span.length != 1:
            raise ValueError("Expected one native instruction layer applied twice")
        projections = [
            (path, m) for path, m in layer.named_modules() if isinstance(m, LoRALinear)
        ]
        if sorted(p for p, _ in projections) != sorted(recipe["projections"]):
            raise ValueError("Incomplete trained task update")
        for path, adapter in projections:
            parent, key = path.rsplit(".", 1)
            # Combine the trained low-rank delta and donor in FP32, with one final rounding.
            linear = adapter.linear
            change = adapter.lora_b.T.astype(mx.float32) @ adapter.lora_a.T.astype(
                mx.float32
            )
            linear.weight = (
                linear.weight.astype(mx.float32) + strength * adapter.scale * change
            ).astype(linear.weight.dtype)
            setattr(getattr(layer, parent), key, linear)
        model.alpha_override = recurrence
        mx.eval(layer.parameters())
        model.eval()
        self._tool_model = model
        cfg = read(self.directory / "config.json")
        self.tokenizer = self._tokenizer(
            self.directory, eos_token_ids=[cfg["text_config"]["eos_token_id"]]
        )
        self.reuse_prefix = True
