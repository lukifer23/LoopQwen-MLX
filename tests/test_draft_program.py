import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.deliberation import ordinary_view
from loopqwen.draft_program import draft, prefill
from loopqwen.tool_model import tool_view
from loopqwen.worker import acquire


def test_native_draft_matches_causal_greedy_tokens_and_excludes_proposed_boundary():
    acquire()
    mx.random.seed(78143)
    args = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        full_attention_interval=1,
        vocab_size=128,
        tie_word_embeddings=False,
    )
    net = TextModel(args)
    net.set_dtype(mx.bfloat16)
    recurrent = tool_view(net, 1, 4, 2)
    ordinary = ordinary_view(recurrent)
    ids = [17, 33, 20, 75, 9]
    expected = []
    for _ in range(4):
        hidden = ordinary.hidden(mx.array([ids + expected]))
        expected.append(int(ordinary.project(hidden[:, -1:]).argmax().item()))
    actual, boundary, first = draft(ordinary, ids, 4, set())
    assert actual == expected and boundary is None and first is not None
    assert draft(ordinary, ids, 4, {expected[0]}) == ([], expected[0], None)
    assert draft(ordinary, ids, 0, set()) == ([], None, None)
    _, h = prefill(recurrent, ids + actual, None)
    full = recurrent.hidden(mx.array([ids + expected]))[:, -1:]
    assert mx.allclose(h, full, atol=0.02, rtol=0.02).item()
