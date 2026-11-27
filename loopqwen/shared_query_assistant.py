"""Load and serve the trained tied-query decoder through the ordinary Qwen tool interface."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .conditional_reentry import resident
from .export_runtime import load_export_tokenizer
from .io import digest, read
from .shared_query import SharedQuery
from .tool_model import Assistant


def from_backbone(directory, folder, spec, identity):
    # Use the same digest-verified Q4 text arrays as the earlier instruction studies.
    donor = resident(directory, folder, spec, identity)
    if spec.get("compute_precision") == "float16":
        donor.text.set_dtype(mx.float16, lambda dtype: dtype == mx.bfloat16)
        mx.eval(donor.text.parameters())
    mx.random.seed(spec["fit_seed"])
    model_class = SharedQuery
    if spec.get("content_read"):
        from .content_read import ContentRead

        model_class = ContentRead
        if spec.get("bounded_write"):
            from .bounded_read import BoundedContentRead

            model_class = BoundedContentRead
            if spec.get("source_copy"):
                from .source_copy import SourceCopyRead

                model_class = SourceCopyRead
    net = model_class(
        donor.text,
        revisits=spec["revisits"],
        rank=spec["rank"],
        gate_rank=spec["gate_rank"],
    )
    net.eval()
    mx.eval(net.parameters())
    return net


class SharedQueryAssistant(Assistant):
    native_plain = True
    default_cache_state_prefill = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        folder = Path(policy)
        spec, fit = read(folder / "registration.json"), read(folder / "fit.json")
        weights = folder / "step.safetensors"
        if (
            not spec.get("shared_query")
            or fit["weights"] != digest(weights)
            or fit["registration"] != digest(folder / "registration.json")
        ):
            raise ValueError("Invalid shared-query weights or registration")
        if (folder / "checksums.json").exists():
            inventory = read(folder / "checksums.json")
            if any(
                inventory[k] != digest(folder / k)
                for k in ("registration.json", "fit.json", weights.name)
            ):
                raise ValueError("Shared-query archive does not match the installation")
        net = from_backbone(directory, folder, spec, fit["backbone"])
        actual = mx.load(str(weights))
        expected = dict(tree_flatten(net.step.parameters()))
        if actual.keys() != expected.keys() or any(
            (v.shape, v.dtype) != (expected[k].shape, expected[k].dtype)
            for k, v in actual.items()
        ):
            raise ValueError(
                "Learned query controller does not fit the registered model"
            )
        net.step.load_weights(list(actual.items()), strict=True)
        mx.eval(net.parameters())
        self._tool_model = net
        self.tokenizer = load_export_tokenizer(
            directory,
            eos_token_ids=[
                read(Path(directory) / "config.json")["text_config"]["eos_token_id"]
            ],
        )
        self.reuse_prefix = True

    def respond(self, question, tools=(), **options):
        options.setdefault("prefill_cache_state", self.default_cache_state_prefill)
        return super().respond(question, tools, **options)
