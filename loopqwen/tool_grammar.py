"""Constrain a model-chosen tool call to the schemas supplied for this request."""

import json
from weakref import WeakKeyDictionary

import mlx.core as mx
import numpy as np

# Tokenizer conversion includes the whole BPE trie. Reuse it on subsequent requests
# with the same tokenizer, and release it when the model is unloaded.
_tokenizers = WeakKeyDictionary()


class ToolGrammar:
    def __init__(self, tokenizer, vocab_size, tools, stop_ids):
        from llguidance import LLMatcher, LLTokenizer, grammar_from

        key = (vocab_size, tuple(sorted(stop_ids)))
        cache = _tokenizers.setdefault(tokenizer, {})
        if key not in cache:
            cache[key] = LLTokenizer(
                tokenizer.backend_tokenizer.to_str(),
                n_vocab=vocab_size,
                eos_token=sorted(stop_ids),
            )
        calls = [
            dict(
                type="object",
                properties=dict(name=dict(const=t.name), arguments=t.parameters),
                required=["name", "arguments"],
                additionalProperties=False,
            )
            for t in tools
        ]
        # An unrestricted text response remains available. Only an opening brace
        # commits to the JSON action branch; ordinary prose keeps its existing decoder.
        grammar = r"""start: WS? (action WS? | TEXT)?
WS: /[ \t\r\n]+/
TEXT: /[^\s{\x00](?s:.)*/
action: %json """ + json.dumps({"anyOf": calls})
        self.matcher = LLMatcher(cache[key], grammar_from("lark", grammar), log_level=0)
        if self.matcher.is_error():
            raise ValueError("Unsupported tool grammar: " + self.matcher.get_error())
        self.end_think = tokenizer.encode("</think>", add_special_tokens=False)
        self.tokenizer = tokenizer
        self.stop_ids = stop_ids
        self.begin_turn()

    def begin_turn(self):
        if not self.matcher.reset():
            raise ValueError(self.matcher.get_error())
        self.thinking = True
        self.prose = False
        self.in_call = False
        self.replaced = 0

    def choose(self, logits, preferred, previous):
        if self.prose:
            return preferred
        if self.thinking:
            n = len(self.end_think)
            self.thinking = (
                [*previous[-(n - 1) :], preferred] if n > 1 else [preferred]
            ) != self.end_think
            return preferred
        token = preferred
        if self.matcher.validate_tokens([token]) != 1:
            if logits is None:
                raise ValueError("Recorded token violates the tool schema")
            mask = np.frombuffer(self.matcher.compute_logit_bias(), dtype=np.uint8)
            mask = mask[: logits.size]
            if mask.size != logits.size or not mask.any():
                raise ValueError("The tool schema has no admissible next token")
            allowed = mx.array(mask) != 0
            token = mx.argmax(mx.where(allowed, logits, -mx.inf)).item()
            self.replaced += 1
        if token in self.stop_ids and self.matcher.is_accepting():
            return token
        if not self.matcher.consume_token(token):
            raise ValueError(self.matcher.get_error())
        if not self.in_call:
            visible = self.tokenizer.decode([token]).lstrip()
            if visible:
                self.in_call = visible.startswith("{")
                self.prose = not self.in_call
        return token
