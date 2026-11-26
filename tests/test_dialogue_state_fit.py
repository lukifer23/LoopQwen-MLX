import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.research.dialogue_state_fit import StateFit, tensors
from loopqwen.tool_model import tool_view
from loopqwen.worker import acquire


def test_supervised_first_pass_is_the_actual_inference_graph():
    acquire()
    mx.random.seed(741)
    args = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        full_attention_interval=4,
        vocab_size=128,
        tie_word_embeddings=True,
    )
    decoder = tool_view(TextModel(args), 3, 8, 2)
    decoder.set_dtype(mx.bfloat16)
    decoder.eval()
    state = StateFit(decoder, 16)
    h = decoder.prefix_hidden(mx.array([[4, 11, 5, 9, 2]]))[0]
    pack = tensors([((h, 3, [3, 8]), [1, 2], [True, False]), ((h, 4, [10]), [], [])])
    h, _, _, _, q, slots, known, valid = pack
    first = decoder._execute(h, [(3, 0)])
    full = decoder._execute(h, [(3, 0), (3, 1)])
    second = decoder._execute(first, [(3, 1)])
    assert mx.array_equal(second, full).item()

    def criterion(m):
        f = m.decoder._execute(h, [(3, 0)])
        logits = m.read_fields(f, q, slots)
        return (
            nn.losses.binary_cross_entropy(
                logits, known, with_logits=True, reduction="none"
            )
            * valid
        ).sum()

    loss, grads = nn.value_and_grad(state, criterion)(state)
    assert loss.item() > 0 and mx.abs(grads["query"]["weight"]).sum().item() > 0
