import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from loopqwen.tool_model import Assistant


def test_same_assistant_reuses_only_identical_tokenizer_inputs(tmp_path):
    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "tool-model.json").write_text(json.dumps({"schema": 2}))

    def save(vocab):
        tokenizer = Tokenizer(
            WordLevel(
                dict(zip(vocab, range(len(vocab)), strict=True)), unk_token="<unk>"
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<end>"
        )
        fast.save_pretrained(tmp_path)

    save(("<unk>", "<end>", "read", "write"))
    assistant = Assistant(tmp_path, reuse_tokenizer=True)
    first = assistant._tokenizer(tmp_path, eos_token_ids=[1])
    assert first is assistant._tokenizer(tmp_path, eos_token_ids=[1])
    assert first.encode("read", add_special_tokens=False) == [2]
    save(("<unk>", "<end>", "write", "read"))
    second = assistant._tokenizer(tmp_path, eos_token_ids=[1])
    assert second is not first and second.encode("read", add_special_tokens=False) == [
        3
    ]
    assert assistant._tokenizer(tmp_path, eos_token_ids=[0]) is not second


def test_qwen_export_preserves_multilingual_graphemes_and_every_frozen_prompt():
    from pathlib import Path

    import pytest
    from mlx_lm.utils import load_tokenizer

    from loopqwen.export_runtime import load_export_tokenizer
    from loopqwen.io import read
    from loopqwen.model_loader import local_snapshot
    from loopqwen.research.shared_session_study import sources, tools_for
    from loopqwen.tool_session import initial_messages, prompt

    spec = read("research/experiments/shared-session-unified-01.json")
    root = Path(spec["checkpoint"])
    if not (root / "tokenizer.json").is_file():
        pytest.skip("Requires the installed release and pinned Base tokenizer")
    base = load_tokenizer(local_snapshot())
    candidate = load_export_tokenizer(root, eos_token_ids=base.eos_token_ids)
    for text in ("समझौता एक्सप्रेस", "ਸਮਝੌਤਾ ਐਕਸਪ੍ਰੈਸ", "கணிதம்", "مَرْحَبًا", "naïve café"):
        assert candidate.encode(text, add_special_tokens=False) == base.encode(
            text, add_special_tokens=False
        )
    field, panel = sources(spec)
    for row in panel:
        tools = tools_for(row, field, True)
        question = row.get("user_request", row["question"])
        messages = (
            initial_messages(question, tools)
            if tools
            else [{"role": "user", "content": question}]
        )
        assert (
            candidate.encode(prompt(candidate, messages), add_special_tokens=False)
            == row["initial_ids"]
        )
