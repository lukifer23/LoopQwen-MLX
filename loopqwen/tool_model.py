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


def with_adapter(model, weights, expected_digest):
    """Create a second recurrent view sharing the immutable donor arrays."""
    if digest(weights) != expected_digest:
        raise ValueError("Observation adapter checksum mismatch")
    shared = {id(v): v for _, v in tree_flatten(model.parameters())}
    view = deepcopy(model, shared)
    tensors = mx.load(str(weights))
    expected = dict(tree_flatten(view.trainable_parameters()))
    if tensors.keys() != expected.keys() or any(
        v.shape != expected[k].shape for k, v in tensors.items()
    ):
        raise ValueError(
            "Observation adapter is incompatible with the first-turn graph"
        )
    view.load_weights(list(tensors.items()), strict=False)
    view.eval()
    return view


def bind(language_model, folder, adapter=None):
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
    expected_digest = meta["weights"]
    if adapter is not None:
        adapter = Path(adapter)
        spec = read(adapter / "registration.json")
        fit = read(adapter / "fit.json")
        identity = ("donor", "prefix_bits", "rank", "start_layer")
        if (
            any(
                spec.get(k, 23 if k == "start_layer" else None) != meta.get(k)
                for k in identity
            )
            or fit["passes"] != meta["passes"]
        ):
            raise ValueError(
                "Tool adapter uses a different backbone, quantization or recurrent graph"
            )
        weights = adapter / "tool-adapters.safetensors"
        expected_digest = fit["weights"]
        inventory = adapter / "checksums.json"
        if inventory.is_file():
            checks = read(inventory)
            if (
                any(
                    digest(adapter / k) != checks[k]
                    for k in ("registration.json", "fit.json")
                )
                or checks[weights.name] != expected_digest
            ):
                raise ValueError("Tool adapter evidence changed")
    if digest(weights) != expected_digest:
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
    def __init__(self, directory, *, tool_adapter=None, reuse_tokenizer=None):
        self.directory = Path(directory)
        self.tool_adapter = Path(tool_adapter) if tool_adapter is not None else None
        meta = read(self.directory / "tool/tool-model.json")
        self.dedicated = meta["schema"] == 2
        policy = meta.get("prefix_reuse")
        if policy is not None and policy["checkpoint"] != meta["weights"]:
            raise ValueError("Prefix policy belongs to different tool weights")
        self.reuse_prefix = policy is not None and (
            self.tool_adapter is None
            or read(self.tool_adapter / "fit.json")["weights"] == meta["weights"]
        )
        self.language = None
        self.tokenizer = None
        self._tool_model = None
        if reuse_tokenizer is None:
            # The complete packed-03 field crossover qualified this tokenizer only.
            inventory = self.directory / "tool-export-files.json"
            reuse_tokenizer = inventory.is_file() and digest(inventory) == (
                "b92ee035e76e64cb05a43166b2a7a9a0d1eea5b87decca964ee9931adc840176"
            )
        self._reuse_tokenizer = reuse_tokenizer
        self._cached_tokenizer = None

    def _tokenizer(self, directory, *, eos_token_ids):
        from .export_runtime import load_export_tokenizer

        if self._reuse_tokenizer:
            # Recheck actual bytes that define the certified Qwen tokenizer. A mode
            # switch can reuse its BPE trie, while replacement files invalidate it.
            names = (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "added_tokens.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "vocab.json",
                "merges.txt",
            )
            files = [directory / n for n in names]
            files.extend(sorted((directory / "chat_templates").glob("*.jinja")))
            key = (
                tuple(eos_token_ids),
                tuple((p.name, digest(p)) for p in files if p.is_file()),
            )
            if self._cached_tokenizer is not None and key == self._cached_tokenizer[0]:
                return self._cached_tokenizer[1]
            tokenizer = load_export_tokenizer(directory, eos_token_ids=eos_token_ids)
            self._cached_tokenizer = key, tokenizer
            return tokenizer
        return load_export_tokenizer(directory, eos_token_ids=eos_token_ids)

    def conversation(self, tools=()):
        """Keep user turns, visible answers and real tool observations across requests."""
        from .tool_session import Conversation

        return Conversation(self, tools)

    def _load_language(self):
        if self.language is None:
            from .export_runtime import load_export

            if self.dedicated:
                self._tool_model = None
                mx.clear_cache()
            if self._reuse_tokenizer:
                self.language, self.tokenizer = load_export(
                    self.directory, _load_tokenizer=self._tokenizer
                )
            else:
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
        record_kv_history=False,
        _messages=None,
    ):
        if tools:
            from .tool_session import run

            if self._tool_model is None:
                if self.dedicated:
                    self.language = None
                    mx.clear_cache()
                    config = read(self.directory / "config.json")
                    self.tokenizer = self._tokenizer(
                        self.directory,
                        eos_token_ids=[config["text_config"]["eos_token_id"]],
                    )
                else:
                    self._load_language()
                self._tool_model = bind(
                    self.language, self.directory / "tool", self.tool_adapter
                )
            return run(
                self._tool_model,
                self.tokenizer,
                question,
                tools,
                max_tokens=max_tokens,
                max_turns=max_turns,
                progress=progress,
                _messages=_messages,
                record_kv_history=record_kv_history,
                reuse_prefix=self.reuse_prefix
                if reuse_prefix is None
                else reuse_prefix,
            )
        from .evaluation.generation_eval import generate
        from .tool_session import prompt

        self._load_language()
        messages = [] if _messages is None else _messages
        messages.append(dict(role="user", content=question))
        text = prompt(self.tokenizer, messages)
        out = generate(
            self.language,
            self.tokenizer,
            text,
            max_tokens,
            stop_ids=[self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
            progress=progress,
            record_kv_history=record_kv_history,
        )
        ids = out["token_ids"]
        turn = dict(
            prompt_ids=self.tokenizer.encode(text, add_special_tokens=False),
            token_ids=ids,
            seconds=out["elapsed_s"],
            ttft_s=out["ttft_s"],
            peak_bytes=out["peak_bytes"],
        )
        if record_kv_history:
            turn.update(
                {
                    k: out[k]
                    for k in (
                        "kv_history",
                        "recurrent_from",
                        "page_sites",
                        "attention_memory",
                    )
                    if k in out
                }
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
        if answer and _messages is not None:
            messages.append(dict(role="assistant", content=answer))
        result = dict(
            answer=answer,
            status=status,
            turns=[turn],
            seconds=out["elapsed_s"],
            context_route=out.get("context_route"),
        )
        if "prefill_policy" in out:
            result["prefill_policy"] = out["prefill_policy"]
        return result
