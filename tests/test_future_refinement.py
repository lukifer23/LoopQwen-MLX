import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.research.future_token_fit import Horizon
from loopqwen.selective_refinement import ProjectedRefinement, SharedRefinement
from loopqwen.worker import acquire


def tiny_text_decoder():
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
    return TextModel(args)


def test_future_supervision_updates_the_causal_shared_block():
    acquire()
    mx.random.seed(603)
    model = SharedRefinement(tiny_text_decoder(), 8, 0.10)
    model.eval()
    objective = Horizon(model, 16)
    ids = mx.array([[3, 17, 6, 21, 11]])
    prefix = model.prefix_hidden(ids)

    def loss(obj):
        first = obj.decoder.ordinary(prefix)
        logits = obj.future_logits(first[:, :-2])
        return nn.losses.cross_entropy(logits, ids[:, 2:]).mean()

    value, grad = nn.value_and_grad(objective, loss)(objective)
    assert value.item() > 0
    assert (
        mx.abs(grad["decoder"]["refinement"]["self_attn"]["v_proj"]["lora_b"])
        .sum()
        .item()
        > 0
    )
    assert mx.abs(grad["up"]["weight"]).sum().item() > 0
    full = model.hidden(ids)
    changed = ids.at[:, -1].add(7)
    assert mx.array_equal(full[:, :-1], model.hidden(changed)[:, :-1]).item()
    cache = model.make_cache()
    mx.eval(model.hidden(ids[:, :-1], cache))
    actual = model.hidden(ids[:, -1:], cache)
    assert cache.slots[3, 0] is not cache.slots[3, 1]
    assert cache.slots[3, 0].offset == cache.slots[3, 1].offset == ids.shape[1]
    assert mx.allclose(actual[:, -1], full[:, -1], atol=1e-4, rtol=1e-4).item()


def test_second_pass_can_learn_a_return_without_changing_the_ordinary_state():
    from loopqwen.research.source_balanced import return_receipt

    acquire()
    mx.random.seed(671)
    model = ProjectedRefinement(tiny_text_decoder(), 8, 0.25, 16)
    model.eval()
    tokens = [3, 17, 6, 21, 11]
    initial = return_receipt(model, tokens)
    assert initial["max_hidden_change"] == 0
    assert all(initial["shared_original_projections"].values())
    ids = mx.array([tokens])
    ordinary = model.ordinary(model.prefix_hidden(ids))
    mx.eval(ordinary)

    def loss(net):
        z = net.project(net.hidden(ids)[:, :-1])
        return nn.losses.cross_entropy(z, ids[:, 1:]).mean()

    _, grad = nn.value_and_grad(model, loss)(model)
    update = grad["return_up"]["weight"]
    assert mx.sum(mx.abs(update)).item() > 0
    model.return_up.weight = model.return_up.weight - 0.05 * update
    assert return_receipt(model, tokens)["max_hidden_change"] > 0
    assert mx.array_equal(ordinary, model.ordinary(model.prefix_hidden(ids))).item()
    cache = model.make_cache()
    mx.eval(model.hidden(ids[:, :-1], cache))
    last = model.hidden(ids[:, -1:], cache)
    assert cache.slots[3, 0] is not cache.slots[3, 1]
    assert mx.allclose(last, model.hidden(ids)[:, -1:], atol=1e-4, rtol=1e-4).item()


def test_joint_answer_objective_differentiates_the_shared_native_attention():
    from functools import partial

    from loopqwen.research.unified_session_fit import joint_loss
    from loopqwen.tool_model import tool_view

    acquire()
    mx.random.seed(830)
    model = tool_view(tiny_text_decoder(), 3, 8, 2)
    model.eval()
    h = model.prefix_hidden(mx.array([[1, 2, 3, 4], [1, 3, 5, 7]]))
    loss = partial(joint_loss, first_weight=0.18, contrast_weight=0.20)
    value, grads = nn.value_and_grad(model, loss)(
        model,
        h,
        mx.array([0, 0, 1]),
        mx.array([2, 3, 2]),
        mx.array([11, 99, 12]),
        mx.array([0.25, 0.25, 0.5]),
        mx.array([0, 2]),
        mx.array([12, 11]),
        mx.array([1.0, 1.0]),
    )
    assert mx.isfinite(value).item()
    from mlx.utils import tree_flatten

    updated = [
        mx.sum(mx.abs(v)).item()
        for k, v in tree_flatten(grads)
        if k.endswith("v_proj.lora_b")
    ]
    assert len(updated) == 1 and updated[0] > 0
