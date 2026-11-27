import mlx.core as mx
import pytest

from test_shared_query import model
from loopqwen.bounded_read import BoundedContentRead
from loopqwen.content_read import ContentRead
from loopqwen.evaluation.generation_eval import generate
from loopqwen.looped_model import LoopedModel
from loopqwen.offloaded_prefill import DocumentPrefix, PrefixOnDisk
from loopqwen.recurrence import Span


class NumericTokenizer:
    eos_token_ids = set()

    @staticmethod
    def encode(prompt, add_special_tokens=False):
        return list(map(int, prompt.split()))

    @staticmethod
    def decode(tokens):
        return " ".join(map(str, tokens))


@pytest.mark.parametrize("kind", ["base", "content", "bounded"])
def test_document_generation_matches_full_openings_and_leaves_shared_prefix_unchanged(
    tmp_path, kind
):
    parent = model()
    text = parent.text
    text.set_dtype(mx.float16)
    reader = BoundedContentRead if kind == "bounded" else ContentRead
    net = (
        LoopedModel(text, Span(len(text.layers) - 1, 1, 1))
        if kind == "base"
        else reader(text, revisits=2, rank=8, gate_rank=4)
    )
    if kind != "base":
        net.step.write.weight = mx.random.normal(net.step.write.weight.shape) * 0.01
    net.eval()
    prefix = [(17 * i + 3) % 128 for i in range(129)]
    root = tmp_path / "source"
    root.mkdir()
    source = DocumentPrefix(net, prefix, root)
    tok = NumericTokenizer()

    def run(tail, reuse):
        prompt = tok.decode(prefix + tail)
        return generate(
            net,
            tok,
            prompt,
            max_tokens=5,
            return_prompt_ids=True,
            record_kv_history=True,
            **(
                {"document_prefix": source}
                if reuse
                else {"staged_context_dir": tmp_path}
            ),
        )

    tails = ([33, 20, 17], [41] * 135)
    for tail in (tails[0], tails[1], tails[0]):
        actual = run(tail, True)
        control = run(tail, False)
        assert actual["prompt_ids"] == control["prompt_ids"] == prefix + tail
        assert actual["token_ids"] == control["token_ids"]
        assert actual["kv_history"] == control["kv_history"]
        assert actual["reused_prompt_tokens"] == len(prefix) and actual[
            "new_prompt_tokens"
        ] == len(tail)
        assert all(
            s.tail.empty() and s.physical_length() == len(prefix)
            for s in source.disk.cache.slots.values()
            if isinstance(s, PrefixOnDisk)
        )
    with pytest.raises(ValueError, match="must extend"):
        generate(
            net,
            tok,
            tok.decode([9] + prefix[1:] + [1]),
            max_tokens=2,
            document_prefix=source,
        )
