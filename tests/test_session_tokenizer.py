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
