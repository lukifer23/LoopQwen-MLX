import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from loopqwen.reserved_cache import ReservedKVCache


def test_fixed_allocation_keeps_values_and_only_exposes_consumed_history():
    fixed, stock = ReservedKVCache(300), KVCache()
    for width in (128, 128, 17, 27):
        pos = mx.arange(fixed.offset, fixed.offset + width)
        k = pos.astype(mx.bfloat16)[None, None, :, None]
        v = mx.stack([pos, 1 + pos], axis=-1).astype(mx.bfloat16)[None, None]
        got = fixed.update_and_fetch(k, v)
        expected = stock.update_and_fetch(k, v)
        assert all(
            mx.array_equal(a, b).item() for a, b in zip(got, expected, strict=True)
        )
        assert fixed.keys.shape[2] == 300 and got[0].shape[2] == stock.offset
    with pytest.raises(ValueError, match="exceeded"):
        fixed.update_and_fetch(k, v)
    assert fixed.offset == 300
