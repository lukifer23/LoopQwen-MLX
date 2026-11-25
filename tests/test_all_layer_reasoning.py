import gc

import mlx.core as mx

from loopqwen.deployment import load_trained
from loopqwen.io import read
from loopqwen.research.all_layer import REG, expanded


def test_zero_initialized_extra_adapters_reproduce_the_trained_parent_logits():
    spec = read(REG)
    parent, tok, _ = load_trained(spec["parent"])
    ids = mx.array([tok.encode("Six boxes contain eight pencils each.")])
    expected = parent(ids)[:, -1, :]
    mx.eval(expected)
    del parent
    gc.collect()
    mx.clear_cache()
    new, _, _ = expanded(spec)
    new.eval()
    assert new.span.passes == 2
    assert mx.array_equal(new(ids)[:, -1, :], expected).item()
