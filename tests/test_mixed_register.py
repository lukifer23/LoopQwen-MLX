import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.looped_model import LoopedModel
from loopqwen.mixed_register import build_view
from loopqwen.recurrence import Span
from loopqwen.worker import acquire


def test_mixed_register_shares_inputs_and_preserves_original_feedforward_arrays():
    acquire()
    mx.random.seed(6349)
    cfg = TextModelArgs(
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
    )
    model = LoopedModel(TextModel(cfg), Span(start=2, length=2, passes=2))
    model.set_dtype(mx.bfloat16)
    model.eval()
    x = mx.array([[1, 5, 10, 4]])
    before = model.project(model.hidden(x))
    mx.eval(before)
    view, receipt = build_view(model)
    assert receipt["shared_input_embedding"] and len(receipt["projections"]) == 12
    assert isinstance(view.text.layers[0].mlp.up_proj, nn.QuantizedLinear)
    assert view.text.layers[0].mlp.up_proj is not model.text.layers[0].mlp.up_proj
    result = view.project(view.hidden(x))
    assert result.shape == before.shape and mx.isfinite(result).all().item()
    assert (
        view.projection_calls == 1
        and mx.array_equal(before, model.project(model.hidden(x))).item()
    )
