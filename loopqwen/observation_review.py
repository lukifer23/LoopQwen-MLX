"""Use a learned second pass after an actual native function response."""

from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from . import REVISION
from .io import digest, read
from .looped_model import LoopedModel
from .recurrence import Span
from .tool_model import Assistant, quantize_prefix, tool_view
from .tool_session import run
from .worker import acquire


def backbone(directory, spec):
    acquire()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("Observation review requires native MLX Metal")
    directory = Path(directory)
    weights = directory / "model.safetensors"
    if (
        read(directory / "provenance.json")["base_revision"] != REVISION
        or digest(weights) != spec["base_text_sha256"]
        or digest(directory / "config.json") != spec["config_sha256"]
    ):
        raise ValueError("Observation review requires the registered Base text weights")
    text = TextModel(
        TextModelArgs.from_dict(read(directory / "config.json")["text_config"])
    )
    text.load_weights(str(weights), strict=True)
    quantize_prefix(text, spec["start_layer"], spec["prefix_bits"])
    mx.eval(text.parameters())
    mx.clear_cache()
    text.eval()
    return text


class ObservationAssistant(Assistant):
    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        self.policy = Path(policy)
        self.spec = read(self.policy / "registration.json")
        self.fit = read(self.policy / "fit.json")
        checks = self.policy / "checksums.json"
        if checks.is_file():
            inventory = read(checks)
            if any(
                digest(self.policy / n) != inventory[n]
                for n in ("registration.json", "fit.json", "tool-adapters.safetensors")
            ):
                raise ValueError("Observation review evidence changed")
        if not self.spec.get("observation_review") or self.fit["passes"] != 2:
            raise ValueError("Invalid review policy")
        self.review = None
        self.reuse_prefix = True

    def load_tools(self):
        if self.review is not None:
            return
        self.language = None
        mx.clear_cache()
        text = backbone(self.directory, self.spec)
        self._tool_model = LoopedModel(text, Span(passes=1))
        shared = {id(v): v for _, v in tree_flatten(text.parameters())}
        self.review = tool_view(
            deepcopy(text, shared), self.spec["start_layer"], self.spec["rank"], 2
        )
        weights = self.policy / "tool-adapters.safetensors"
        if digest(weights) != self.fit["weights"]:
            raise ValueError("Review adapter checksum mismatch")
        tensors = mx.load(str(weights))
        shape = dict(tree_flatten(self.review.trainable_parameters()))
        if tensors.keys() != shape.keys() or any(
            a.shape != shape[n].shape for n, a in tensors.items()
        ):
            raise ValueError("Review adapter tensor set differs")
        self.review.load_weights(list(tensors.items()), strict=False)
        self.review.eval()
        self._tool_model.eval()
        self.tokenizer = self._tokenizer(
            self.directory,
            eos_token_ids=[
                read(self.directory / "config.json")["text_config"]["eos_token_id"]
            ],
        )

    def respond(self, question, tools=(), **options):
        if not tools:
            self.review = self._tool_model = None
            return super().respond(question, **options)
        self.load_tools()
        if options.get("reuse_prefix") is None:
            options["reuse_prefix"] = self.reuse_prefix
        options.setdefault("max_tokens", 160)
        return run(
            self._tool_model,
            self.tokenizer,
            question,
            tools,
            observation_model=self.review,
            tool_format="qwen",
            **options,
        )
