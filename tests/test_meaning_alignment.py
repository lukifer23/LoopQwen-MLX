import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.meaning_alignment import AlignedReader, PairRequest
from loopqwen.tool_model import tool_view
from loopqwen.worker import acquire


def test_native_alignment_matches_training_and_cannot_change_an_earlier_token():
    acquire()
    mx.random.seed(3193)
    config = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        full_attention_interval=1,
        vocab_size=128,
        tie_word_embeddings=True,
    )
    text = TextModel(config)
    text.set_dtype(mx.bfloat16)
    reader = AlignedReader(tool_view(text, 1, 4, 2), 16)
    reader.eval()
    ids = mx.array([[4, 37, 15, 14, 21, 17, 7, 18, 28, 11, 20, 32]])
    spans = [[1, 4], [5, 9]]
    end_prompt = 10
    h = reader.decoder.prefix_hidden(ids)
    left, right = [h[:, a:b] for a, b in spans]
    lm, rm = [mx.ones(v.shape[:2], mx.bool_) for v in (left, right)]
    relation = reader.relation(left, right, lm, rm)
    padded = reader.relation(
        mx.pad(left, ((0, 0), (0, 4), (0, 0))), right, mx.pad(lm, ((0, 0), (0, 4))), rm
    )
    assert mx.allclose(relation, padded, atol=1e-4).item()
    revised = reader.condition(
        h, relation, mx.arange(ids.shape[1])[None, :] >= end_prompt - 1
    )
    assert mx.array_equal(h[:, :9], revised[:, :9]).item()
    assert mx.max(mx.abs(h[:, 9:] - revised[:, 9:])).item() > 0
    expected = reader.read_suffix(revised)
    for width in (2, 7, 12):
        view = PairRequest(reader, spans, end_prompt)
        cache = view.make_cache()
        chunks = [
            view.hidden(ids[:, i : i + width], cache) for i in range(0, 12, width)
        ]
        assert mx.allclose(
            mx.concatenate(chunks, 1), expected, atol=0.045, rtol=0.02
        ).item()
