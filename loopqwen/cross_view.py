"""Serve the sealed instruction backbone with its compiled recurrent reading layer."""

from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from .hardware import require_metal
from .interpolated_instruction import InterpolatedAssistant
from .io import digest, read, save
from .looped_model import LoopedModel
from .recurrence import Span
from .tool_model import Assistant, quantize_prefix
from .worker import acquire


def compile_layer(directory, policy):
    start = perf_counter()
    spec = read(policy / "registration.json")
    actor = InterpolatedAssistant(directory, policy)
    layer = actor._tool_model.text.layers[spec["start_layer"]]
    modules = dict(layer.named_modules())
    paths = spec["instruction_interpolation"]["projections"]
    weights = {
        f"model.layers.{spec['start_layer']}.{p}.weight": modules[p].weight
        for p in paths
    }
    output = policy / "cross-view.safetensors"
    mx.save_safetensors(str(output), weights)
    save(
        policy / "compiled.json",
        {
            "weights": digest(output),
            "adapter": read(policy / "fit.json")["weights"],
            "registration": digest(policy / "registration.json"),
            "projections": sorted(weights),
            "bytes": output.stat().st_size,
            "peak_bytes": mx.get_peak_memory(),
            "seconds": perf_counter() - start,
        },
    )


class CrossViewAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        require_metal()
        acquire()
        directory, policy = Path(directory), Path(policy)
        spec = read(policy / "registration.json")
        recipe = spec["instruction_interpolation"]
        compiled = read(policy / "compiled.json")
        fit = read(policy / "fit.json")
        weights = policy / "cross-view.safetensors"
        folder = directory / "tool"
        meta = read(folder / "tool-model.json")
        if (
            not spec.get("trained_instruction_adapter")
            or (spec["start_layer"], spec["prefix_bits"], fit["passes"]) != (23, 8, 2)
            or compiled["registration"] != digest(policy / "registration.json")
            or compiled["adapter"] != fit["weights"]
            or digest(weights) != compiled["weights"]
            or digest(folder / "tool-model.json") != recipe["metadata_sha256"]
        ):
            raise ValueError("Cross-view recurrent checkpoint identity changed")
        if spec.get("merged_task_updates"):
            manifest = policy / "task-merge.json"
            receipt = read(manifest)
            if (
                digest(manifest) != compiled.get("merge")
                or receipt["compiled"] != compiled["weights"]
                or receipt["new_expert"] != fit["weights"]
                or receipt["parents"] != spec["merged_task_updates"]["parents"]
                or receipt["density"] != spec["merged_task_updates"]["density"]
            ):
                raise ValueError("Merged recurrent projections differ from their sources")
        if (policy / "checksums.json").exists():
            inventory = read(policy / "checksums.json")
            if any(
                digest(policy / k) != inventory[k]
                for k in (
                    "registration.json",
                    "fit.json",
                    "compiled.json",
                    "cross-view.safetensors",
                )
            ):
                raise ValueError("Cross-view publication differs from its archive")
        backbone = folder / "quantized-backbone.safetensors"
        if digest(backbone) != meta["backbone_sha256"]:
            raise ValueError("Different instruction backbone")
        config = read(directory / "config.json")["text_config"]
        text = TextModel(TextModelArgs.from_dict(config))
        quantize_prefix(text, 23, 8)
        tensors = mx.load(str(backbone))
        fused = mx.load(str(weights))
        expected = sorted(
            "model.layers.23." + name + ".weight" for name in recipe["projections"]
        )
        if (
            sorted(fused) != expected
            or compiled["projections"] != expected
            or any(
                k not in tensors
                or v.shape != tensors[k].shape
                or v.dtype != tensors[k].dtype
                for k, v in fused.items()
            )
        ):
            raise ValueError(
                "The compiled recurrent layer differs from its source graph"
            )
        tensors.update(fused)
        text.load_weights(list(tensors.items()), strict=True)
        model = LoopedModel(
            text,
            Span(start=23, length=1, passes=2, mode="stabilized"),
            alpha=recipe["second_pass_fraction"],
        )
        model.alpha_override = recipe["second_pass_fraction"]
        mx.eval(model.parameters())
        model.eval()
        self._tool_model = model
        self.tokenizer = self._tokenizer(
            directory, eos_token_ids=[config["eos_token_id"]]
        )
        self.reuse_prefix = True
