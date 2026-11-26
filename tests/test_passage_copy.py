import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.passage_copy import CopyReader, PassageRequest
from loopqwen.tool_model import tool_view
from loopqwen.worker import acquire


def test_causal_copy_distribution_matches_gold_mass_for_repeated_source_tokens():
    acquire()
    mx.random.seed(637231)
    cfg = TextModelArgs(
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
    text = TextModel(cfg)
    text.set_dtype(mx.bfloat16)
    reader = CopyReader(tool_view(text, 1, 4, 2), 16)
    reader.eval()
    ids = mx.array([[1, 8, 7, 8, 12, 9]])
    outputs = []
    for width in (6, 2):
        view = PassageRequest(reader, (1, 5))
        cache = view.make_cache()
        for start in range(0, 6, width):
            h = view.hidden(ids[:, start : start + width], cache)
        last = h[:, -1:]
        distribution = mx.exp(view.project(last))[0, 0]
        keys, source, previous = view.memory
        mass, g = reader.probabilities(
            last, keys, ids[:, -1:], previous, mx.ones(source.shape, mx.bool_)
        )
        ordinary = mx.softmax(reader.decoder.project(last).astype(mx.float32), axis=-1)[
            0, 0
        ]
        expected = (1 - g.item()) * ordinary[8] + g.item() * (
            mass[0, 0, 0] + mass[0, 0, 2]
        )
        assert source.tolist() == [[8, 7, 8, 12]]
        assert abs(distribution.sum().item() - 1) < 1e-5
        assert abs((distribution[8] - expected).item()) < 1e-6
        outputs.append(distribution)
    assert mx.allclose(outputs[0], outputs[1], rtol=0.025, atol=0.002).item()
