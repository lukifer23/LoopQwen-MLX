"""One recurrent decoder for short reasoning, ordinary replies and real tool calls."""

from pathlib import Path

import mlx.core as mx

from .io import digest, read
from .tool_model import Assistant, bind
from .tool_session import prompt


class UnifiedAssistant(Assistant):
    def __init__(self, directory, policy):
        self.policy = Path(policy)
        spec = read(self.policy / "registration.json")
        if (
            digest(Path(directory) / "tool-export-files.json")
            != spec["bundle_inventory"]
        ):
            raise ValueError("Joint policy requires its registered parent release")
        super().__init__(directory, tool_adapter=self.policy)
        config = read(self.directory / "config.json")
        self.tokenizer = self._tokenizer(
            self.directory, eos_token_ids=[config["text_config"]["eos_token_id"]]
        )
        self.long_threshold = spec["long_threshold"]
        self.reuse_prefix = False
        self._parent = None

    def _load_language(self):
        if self.language is None:
            self._parent = None
            mx.clear_cache()
            self.language = self._tool_model = bind(
                None, self.directory / "tool", self.policy
            )

    def respond(self, question, tools=(), **options):
        if not tools:
            from .evidence_loop import passage

            messages = list(options.get("_messages") or ()) + [
                {"role": "user", "content": question}
            ]
            ids = self.tokenizer.encode(
                prompt(self.tokenizer, messages), add_special_tokens=False
            )
            section = passage(self.tokenizer, ids) if len(ids) <= 1024 else None
            evidence = section is not None and len(section[0]) <= 64
            if evidence or len(ids) >= self.long_threshold:
                if self._parent is None:
                    self.language = self._tool_model = None
                    mx.clear_cache()
                    self._parent = Assistant(self.directory)
                return {
                    **self._parent.respond(question, tools, **options),
                    "short_policy_route": "parent_evidence"
                    if evidence
                    else "parent_long",
                }
        self._load_language()
        return {
            **super().respond(question, tools, **options),
            "short_policy_route": "unified",
        }
