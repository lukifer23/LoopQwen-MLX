"""Check conditional JSON constraints against the production tokenizer's real merges."""

import pytest
from mlx_lm.utils import load_tokenizer

from loopqwen.io import read
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_grammar import ToolGrammar
from loopqwen.tool_session import Tool


def test_caller_schema_only_constrains_json_after_thinking():
    pytest.importorskip("llguidance")
    root = local_snapshot()
    tokenizer = load_tokenizer(root)
    stop_ids = tokenizer.eos_token_ids | {tokenizer.convert_tokens_to_ids("<|im_end|>")}
    item = Tool(
        "lookup",
        "Read flags.",
        {
            "type": "object",
            "properties": {"levels": {"type": "array", "items": {"type": "integer"}}},
            "required": ["levels"],
            "additionalProperties": False,
        },
        lambda levels: levels,
    )
    grammar = ToolGrammar(
        tokenizer,
        read(root / "config.json")["text_config"]["vocab_size"],
        [item],
        stop_ids,
    )

    def replay(text):
        grammar.begin_turn()
        seen = []
        for tok in [*tokenizer.encode(text, add_special_tokens=False), min(stop_ids)]:
            assert grammar.choose(None, tok, seen) == tok
            seen.append(tok)

    replay(
        'Maybe {"name":"other"}</think>\n\n{"name":"lookup","arguments":{"levels":[0,-3]}}'
    )
    replay("</think>\n\nThe result is café {three}.")
    for invalid in (
        '{"name":"other","arguments":{"levels":[]}}',
        '{"name":"lookup","arguments":{"levels":["three"]}}',
        '{"name":"lookup","arguments":{}}',
    ):
        with pytest.raises(ValueError, match="violates the tool schema"):
            replay("</think>\n\n" + invalid)
