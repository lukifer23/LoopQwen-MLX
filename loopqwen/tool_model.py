"""A recurrent tool view that shares the ordinary branch's pinned backbone arrays."""

from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from . import REVISION
from .io import digest, read
from .looped_model import LoopedModel
from .recurrence import Span
from .training.adapters import attach


def tool_view(text, start, rank, passes):
    model = LoopedModel(
        text,
        Span(
            start=start,
            length=len(text.layers) - start,
            passes=passes,
            mode="stabilized",
        ),
        alpha=0.35,
    )
    attach(model, rank=rank, alpha=2 * rank)
    model.freeze(keys=["alpha_logit"], recurse=False)
    return model


def bind(language_model, folder):
    """Install a tool view without mutating the decoder used by plain-text requests."""
    folder = Path(folder)
    meta = read(folder / "tool-model.json")
    if meta["base_revision"] != REVISION or meta["schema"] != 1 or meta["passes"] < 2:
        raise ValueError("Incompatible recurrent tool model")
    weights = folder / "tool-adapters.safetensors"
    if digest(weights) != meta["weights"]:
        raise ValueError("Tool adapter checksum mismatch")
    ordinary = language_model.context_policy.ordinary.text
    shared = {id(v): v for _, v in tree_flatten(ordinary.parameters())}
    view = tool_view(
        deepcopy(ordinary, shared), meta["start_layer"], meta["rank"], meta["passes"]
    )
    tensors = mx.load(str(weights))
    expected = dict(tree_flatten(view.trainable_parameters()))
    if tensors.keys() != expected.keys() or any(
        t.shape != expected[k].shape for k, t in tensors.items()
    ):
        raise ValueError("Tool adapter tensor set mismatch")
    view.load_weights(list(tensors.items()), strict=False)
    view.eval()
    return view


class Assistant:
    def __init__(self, directory):
        from .export_runtime import load_export

        self.directory = Path(directory)
        self.language, self.tokenizer = load_export(directory)
        self._tool_model = None

    def respond(self, question, tools=(), *, max_tokens=160, max_turns=4):
        if tools:
            from .tool_session import run

            if self._tool_model is None:
                self._tool_model = bind(self.language, self.directory / "tool")
            return run(
                self._tool_model,
                self.tokenizer,
                question,
                tools,
                max_tokens=max_tokens,
                max_turns=max_turns,
            )
        from .evaluation.generation_eval import generate
        from .tool_session import prompt

        text = prompt(self.tokenizer, [dict(role="user", content=question)])
        return generate(
            self.language,
            self.tokenizer,
            text,
            max_tokens,
            stop_ids=[self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
        )
