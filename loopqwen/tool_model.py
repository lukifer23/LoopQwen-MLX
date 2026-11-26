"""A recurrent tool view that shares the ordinary branch's pinned backbone arrays."""

from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from . import REVISION
from .io import digest, read
from .looped_model import LoopedModel
from .recurrence import Span
from .training.adapters import attach


def quantize_prefix(text, start, bits):
    def frozen_projection(path, module):
        suffix = any(
            path.startswith(f"model.layers.{i}.")
            for i in range(start, len(text.layers))
        )
        return not suffix and isinstance(module, (nn.Embedding, nn.Linear))

    nn.quantize(text, group_size=64, bits=bits, class_predicate=frozen_projection)


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
    if (
        meta["base_revision"] != REVISION
        or meta["schema"] not in (1, 2)
        or meta["passes"] < 2
    ):
        raise ValueError("Incompatible recurrent tool model")
    weights = folder / "tool-adapters.safetensors"
    if digest(weights) != meta["weights"]:
        raise ValueError("Tool adapter checksum mismatch")
    if meta["schema"] == 2:
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        from .worker import acquire

        acquire()
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            raise RuntimeError("The tool decoder requires native MLX Metal")
        backbone = folder / "quantized-backbone.safetensors"
        if digest(backbone) != meta["backbone_sha256"]:
            raise ValueError("Quantized tool backbone differs")
        cfg = read(folder.parent / "config.json")
        text = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
        quantize_prefix(text, meta["start_layer"], meta["prefix_bits"])
        text.load_weights(str(backbone), strict=True)
        mx.eval(text.parameters())
    else:
        ordinary = language_model.context_policy.ordinary.text
        shared = {id(v): v for _, v in tree_flatten(ordinary.parameters())}
        text = deepcopy(ordinary, shared)
    view = tool_view(text, meta["start_layer"], meta["rank"], meta["passes"])
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
        self.directory = Path(directory)
        meta = read(self.directory / "tool/tool-model.json")
        self.dedicated = meta["schema"] == 2
        policy = meta.get("prefix_reuse")
        if policy is not None and policy["checkpoint"] != meta["weights"]:
            raise ValueError("Prefix policy belongs to different tool weights")
        self.reuse_prefix = policy is not None
        self.language = None
        self.tokenizer = None
        self._tool_model = None

    def _load_language(self):
        if self.language is None:
            from .export_runtime import load_export

            if self.dedicated:
                self._tool_model = None
                mx.clear_cache()
            self.language, self.tokenizer = load_export(self.directory)

    def respond(
        self,
        question,
        tools=(),
        *,
        max_tokens=160,
        max_turns=4,
        progress=None,
        reuse_prefix=None,
    ):
        if tools:
            from .tool_session import run

            if self._tool_model is None:
                if self.dedicated:
                    from mlx_lm.tokenizer_utils import load as load_tokenizer

                    self.language = None
                    mx.clear_cache()
                    config = read(self.directory / "config.json")
                    self.tokenizer = load_tokenizer(
                        self.directory,
                        eos_token_ids=[config["text_config"]["eos_token_id"]],
                    )
                else:
                    self._load_language()
                self._tool_model = bind(self.language, self.directory / "tool")
            return run(
                self._tool_model,
                self.tokenizer,
                question,
                tools,
                max_tokens=max_tokens,
                max_turns=max_turns,
                progress=progress,
                reuse_prefix=self.reuse_prefix
                if reuse_prefix is None
                else reuse_prefix,
            )
        from .evaluation.generation_eval import generate
        from .tool_session import prompt

        self._load_language()
        text = prompt(self.tokenizer, [dict(role="user", content=question)])
        out = generate(
            self.language,
            self.tokenizer,
            text,
            max_tokens,
            stop_ids=[self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
            progress=progress,
        )
        ids = out["token_ids"]
        turn = dict(
            prompt_ids=self.tokenizer.encode(text, add_special_tokens=False),
            token_ids=ids,
            seconds=out["elapsed_s"],
            ttft_s=out["ttft_s"],
            peak_bytes=out["peak_bytes"],
        )
        answer = None
        status = "token_limit"
        if ids[-1] in {
            *self.tokenizer.eos_token_ids,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }:
            _, closed, visible = self.tokenizer.decode(ids[:-1]).partition("</think>")
            if closed:
                answer = visible.strip()
                turn["visible"] = answer
                status = "answered" if answer else "empty_answer"
            else:
                status = "unfinished_thinking"
        return dict(
            answer=answer,
            status=status,
            turns=[turn],
            seconds=out["elapsed_s"],
            context_route=out.get("context_route"),
        )
