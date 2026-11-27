from dataclasses import replace

import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.looped_model import LoopedModel
from loopqwen.offloaded_prefill import OffloadedPrefill
from loopqwen.recurrence import Span
from loopqwen.worker import acquire


@pytest.mark.parametrize("mode", ["direct", "stabilized", "input_anchor"])
@pytest.mark.parametrize("group_attention", [False, True])
def test_disk_prefill_preserves_full_logits_across_three_passes(tmp_path, mode, group_attention):
    acquire()
    mx.random.seed(1701)
    config = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        full_attention_interval=2,
        vocab_size=256,
        tie_word_embeddings=True,
    )
    text = TextModel(config)
    text.set_dtype(mx.bfloat16)
    m = LoopedModel(text, Span(start=1, length=2, passes=3, mode=mode))
    m.eval()
    ids = [(i * 17) % 256 for i in range(259)]
    cache = m.make_cache()
    for first in range(0, len(ids), 128):
        h = m.hidden(mx.array([ids[first : first + 128]]), cache)
        mx.eval(h)
    reference = m.project(h[:, -1:])
    disk = OffloadedPrefill(m, tmp_path, group_attention=group_attention)
    actual = disk.prefill(ids)
    assert len(list(tmp_path.glob("*.safetensors"))) == 4
    for _ in range(3):
        assert mx.array_equal(actual, reference).item()
        token = mx.argmax(reference[:, -1], axis=-1)[:, None]
        reference = m.project(m.hidden(token, cache))
        actual = disk.decode(token)
    assert mx.array_equal(actual, reference).item()
    m.span = replace(m.span, passes=1)
    with pytest.raises(ValueError, match="original model"):
        disk.decode(token)
