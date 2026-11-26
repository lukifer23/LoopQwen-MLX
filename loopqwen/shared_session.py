"""Serve reasoning and tool adapters on one resident instruction decoder."""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from .io import digest, read
from .tool_model import Assistant, bind, with_adapter
from .tool_session import prompt, run, system_message


def pooled(embedding, tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty routing input")
    if len(ids) > 128:
        ids = [ids[i * (len(ids) - 1) // 127] for i in range(128)]
    h = embedding(mx.array(ids)).astype(mx.float32).mean(axis=0)
    return h * mx.rsqrt((h * h).mean() + 1e-6)


def evidence_features(embedding, tokenizer, question, schema):
    user = pooled(embedding, tokenizer, question)
    catalog = pooled(embedding, tokenizer, schema)
    return mx.concatenate([user, catalog, user * catalog])


class EvidenceGate(nn.Module):
    def __init__(self, size, width):
        super().__init__()
        self.down = nn.Linear(size * 3, width)
        self.score = nn.Linear(width, 1)

    def __call__(self, x):
        return self.score(nn.silu(self.down(x)))[..., 0]


class SharedAssistant(Assistant):
    def __init__(self, directory, policy):
        super().__init__(directory)
        self.policy = Path(policy)
        spec = read(self.policy / "registration.json")
        self.fit = read(self.policy / "fit.json")
        self.routing = read(self.policy / "router.json")
        if (
            digest(self.directory / "tool-export-files.json")
            != spec["bundle_inventory"]
        ):
            raise ValueError("Shared policy requires its verified parent export")
        checks = self.policy / "checksums.json"
        if checks.exists():
            inventory = read(checks)
            for name in (
                "registration.json",
                "fit.json",
                "router.json",
                "router.safetensors",
                "tool-adapters.safetensors",
            ):
                if digest(self.policy / name) != inventory[name]:
                    raise ValueError("Shared policy file differs: " + name)
        self.gate_width = spec["gate_width"]
        self.plain_cutoff = spec["plain_cutoff"]
        self.long_threshold = spec["long_threshold"]
        self._long = None
        self.reuse_prefix = False
        self.gate = None
        config = read(self.directory / "config.json")
        self.tokenizer = self._tokenizer(
            self.directory, eos_token_ids=[config["text_config"]["eos_token_id"]]
        )

    def _load_language(self):
        if self.language is not None:
            return
        self._long = None
        mx.clear_cache()
        self._tool_model = bind(None, self.directory / "tool")
        self.language = with_adapter(
            self._tool_model,
            self.policy / "tool-adapters.safetensors",
            self.fit["weights"],
        )
        path = self.policy / "router.safetensors"
        if digest(path) != self.routing["weights"]:
            raise ValueError("Evidence router identity changed")
        self.gate = EvidenceGate(self.language.args.hidden_size, self.gate_width)
        self.gate.load_weights(str(path), strict=True)
        self.gate.eval()
        mx.eval(self.gate.parameters())

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
        options = {
            "max_tokens": max_tokens,
            "max_turns": max_turns,
            "progress": progress,
            "record_kv_history": record_kv_history,
            "reuse_prefix": reuse_prefix,
            "_messages": _messages,
        }
        if not tools:
            msgs = list(_messages or ()) + [{"role": "user", "content": question}]
            ids = self.tokenizer.encode(
                prompt(self.tokenizer, msgs), add_special_tokens=False
            )
            from .evidence_loop import passage

            section = passage(self.tokenizer, ids) if len(ids) <= 1024 else None
            previous_reader = section is not None and len(section[0]) <= 64
            if len(ids) >= self.long_threshold or previous_reader:
                if self._long is None:
                    self.language = self._tool_model = self.gate = None
                    mx.clear_cache()
                    self._long = Assistant(self.directory)
                return {
                    **self._long.respond(question, **options),
                    "short_policy_route": "parent_evidence"
                    if previous_reader
                    else "parent_long",
                }
            self._load_language()
            return {
                **super().respond(question, **options),
                "short_policy_route": "reasoning",
            }
        self._load_language()
        # The serving decision sees only caller-visible text. The callbacks remain
        # available to either decoder, with the same schemas and turn limits.
        summary = question
        if _messages and len(_messages) > 1:
            earlier = [m["content"] for m in _messages if m["role"] == "user"]
            summary = "\n".join([*earlier[-2:], question])
        x = evidence_features(
            self._tool_model.text.model.embed_tokens,
            self.tokenizer,
            summary,
            system_message(tools)["content"],
        )
        external = mx.sigmoid(self.gate(x)).item()
        reasoning = external < self.plain_cutoff
        model = self.language if reasoning else self._tool_model
        options["reuse_prefix"] = (
            self.reuse_prefix if reuse_prefix is None else reuse_prefix
        )
        result = run(model, self.tokenizer, question, tools, **options)
        return {
            **result,
            "short_policy_route": "reasoning" if reasoning else "tool",
            "external_probability": external,
        }
