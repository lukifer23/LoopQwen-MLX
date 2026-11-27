import mlx.core as mx
import pytest
from test_shared_query import model

from loopqwen.long_training_memory import DocumentMemory
from loopqwen.research.shared_query_features import arrays


def test_document_branches_keep_full_memory_without_mutating_source(tmp_path):
    net = model()
    net.text.set_dtype(mx.float16)
    base = [(i * 7 + 2) % 128 for i in range(128)]
    memory = DocumentMemory(net, base, tmp_path)

    def branch(question, labels):
        state = memory.branch(base + question, labels)
        prefix = mx.load(str(state["prefix_file"]))
        full = arrays(net, base + question, labels)
        for name, old, extra in (
            ("keys", "k", "tail_keys"),
            ("values", "v", "tail_values"),
        ):
            assert mx.allclose(
                mx.concatenate([prefix[old], state[extra]], axis=2),
                full[name],
                rtol=0.006,
                atol=0.006,
            ).item()
        assert state["offset"] == len(base + question) - 1 and state[
            "prefix_length"
        ] == len(base)
        assert mx.allclose(state["pre"], full["pre"], rtol=0.006, atol=0.006).item()
        assert mx.allclose(
            state["ordinary"], full["ordinary"], rtol=0.006, atol=0.006
        ).item()
        return state["pre"], state["ordinary"]

    a = branch([13, 14], [16, 52, 38])
    branch([7] * 129, [4, 33, 81])
    replay = branch([13, 14], [16, 52, 38])
    assert all(mx.array_equal(x, y).item() for x, y in zip(a, replay, strict=True))
    with pytest.raises(ValueError, match="exact original"):
        memory.branch([0] + base[1:] + [13, 14], [16])
