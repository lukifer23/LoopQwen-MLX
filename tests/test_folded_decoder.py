import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.folded_decoder import ResidualRepair, fold
from loopqwen.tool_model import tool_view
from loopqwen.worker import acquire


def test_fold_reuses_one_physical_delta_layer_and_keeps_independent_causal_states():
    acquire()
    mx.random.seed(8772)
    cfg = TextModelArgs(
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
        full_attention_interval=4,
        vocab_size=256,
        tie_word_embeddings=True,
    )
    net = TextModel(cfg)
    net.set_dtype(mx.bfloat16)
    tool = tool_view(net, 3, 4, 2)
    plain, tools, removed = fold(tool, ResidualRepair(64, 16), 0)
    assert (
        len(plain.text.layers) == 3 and len(plain.sites) == 4 and len(tools.sites) == 5
    )
    assert removed is not plain.text.layers[0]
    assert all(
        a is b
        for (_, a), (_, b) in zip(
            tree_flatten(plain.text.layers[0].parameters()),
            tree_flatten(tools.text.layers[0].parameters()),
            strict=True,
        )
    )
    ids = mx.array([[3, 11, 9, 8, 25, 51, 60]])
    for model in (plain, tools):
        state = model.make_cache()
        assert state.slots[0, 0] is not state.slots[0, 1]
        full = model.hidden(ids)[:, -1:]
        mx.eval(model.hidden(ids[:, :-1], state))
        cached = model.hidden(ids[:, -1:], state)
        assert mx.allclose(full, cached, rtol=0.035, atol=0.035).item()
