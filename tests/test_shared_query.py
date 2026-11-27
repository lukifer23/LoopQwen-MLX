import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from test_future_refinement import tiny_text_decoder

from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.shared_query import SharedQuery
from loopqwen.worker import acquire


def model(revisits=2):
    acquire()
    mx.random.seed(864371)
    net = SharedQuery(tiny_text_decoder(), revisits=revisits, rank=8, gate_rank=4)
    net.eval()
    return net


def test_shared_query_preserves_stock_at_start_and_writes_history_once():
    net = model()
    stock = LoopedModel(net.text, Span(3, 1, 1))
    stock.eval()
    ids = mx.array([[3, 8, 13, 2, 34, 9, 17, 11, 28]])
    whole = stock(ids)
    assert mx.allclose(net(ids), whole, rtol=1e-5, atol=1e-5).item()
    cache = net.make_cache()
    net.last_hidden(ids[:, :4], cache)
    net.last_hidden(ids[:, 4:7], cache)
    out = net.project(net.last_hidden(ids[:, 7:], cache))
    assert mx.allclose(out, whole[:, -1:], rtol=1e-4, atol=1e-4).item()
    slot = cache.slots[(3, 0)]
    assert (
        slot.offset == ids.shape[1]
        and slot.keys_and_values()[0].shape[2] == ids.shape[1]
    )
    assert len(cache.slots) == len(net.text.layers)


def test_trained_shared_queries_are_causal_and_read_the_actual_values_each_visit():
    net = model()
    net.step.bias = mx.full(net.step.bias.shape, 0.22)
    net.step.content.up.weight = (
        mx.random.normal(net.step.content.up.weight.shape) * 0.04
    )
    ids = mx.array([[3, 8, 13, 2, 34, 9, 17, 11, 28]])
    pre, h, k, v, pos, mask = net.encode(ids)
    exits = []
    r = net.refine(pre, h, k, v, pos, mask, observer=lambda _, n, x: exits.append(x))
    assert len(exits) == 2 and all(
        not mx.allclose(a, b).item() for a, b in ((exits[0], h), (exits[0], exits[1]))
    )
    # Change memory after encoding. Ordinary hidden and all controller weights stay fixed.
    assert not mx.allclose(r, net.refine(pre, h, k, mx.zeros_like(v), pos, mask)).item()
    changed = mx.concatenate((ids[:, :5], mx.array([[11, 2, 7, 93]])), axis=1)
    assert mx.allclose(
        net(ids)[:, :5], net(changed)[:, :5], rtol=1e-5, atol=1e-5
    ).item()
    cache = net.make_cache()
    net.last_hidden(ids[:, :4], cache)
    cached = net.last_hidden(ids[:, 4:], cache)
    assert mx.allclose(cached, net.hidden(ids)[:, -1:], rtol=1e-4, atol=1e-4).item()


def test_backpropagation_updates_the_tied_query_step_and_dense_control_has_equal_capacity():
    net = model()
    ids = mx.array([[3, 8, 13, 2, 34, 9, 17, 11, 28, 5, 32, 12, 34, 3, 4, 2, 30]])
    pre, h, k, v, pos, mask = net.encode(ids[:, :-1])
    net.step.bias = mx.full(net.step.bias.shape, 0.15)
    net.step.content.up.weight = (
        mx.random.normal(net.step.content.up.weight.shape) * 0.04
    )

    def loss(step):
        states = net.refine(pre, h, k, v, pos, mask, step=step, differentiate=True)
        return nn.losses.cross_entropy(
            net.project(net.text.model.norm(states)), ids[:, 1:], reduction="mean"
        )

    cost, grad = nn.value_and_grad(net.step, loss)(net.step)
    assert mx.isfinite(cost).item()
    assert all(mx.isfinite(x).all().item() for _, x in tree_flatten(grad))
    assert mx.linalg.norm(grad["content"]["down"]["weight"]).item() > 0
    assert mx.linalg.norm(grad["bias"]).item() > 0
    one = model(0)

    def sizes(m):
        return {k: v.shape for k, v in tree_flatten(m.trainable_parameters())}

    assert sizes(net) == sizes(one) and all(k.startswith("step.") for k in sizes(net))


def test_native_and_differentiable_attention_agree_for_single_and_partial_suffix():
    from mlx_lm.models.base import scaled_dot_product_attention

    from loopqwen.shared_query import differentiable_attention

    net = model()
    for length in (1, 3, 16):
        k = mx.random.normal((2, 2, 25, 32))
        v = mx.random.normal(k.shape)
        q = mx.random.normal((2, 8, length, 32))
        for mask in (None, "causal"):
            cross = differentiable_attention(q, k, v, 32**-0.5, mask)
            native = scaled_dot_product_attention(
                q, k, v, cache=None, scale=32**-0.5, mask=mask
            )
            assert mx.allclose(cross, native, rtol=2e-5, atol=2e-5).item()
    assert net.revisits == 2


def test_cited_probability_uses_only_causal_keys_and_trains_the_actual_query():
    import math

    from loopqwen.research.shared_query_fit import cited_loss, loss_fn

    net = model()
    q = mx.zeros((1, 2, 3, 32))
    k = mx.zeros((1, 1, 9, 32))
    # The second query can attend to eight keys, two of which are cited.
    assert abs(cited_loss(q, k, 6, [(1, [[0, 2]])], 0.2).item() - math.log(4)) < 1e-5
    ids = mx.array([[3, 8, 13, 2, 34, 9, 17, 11, 28]])
    pre, h, k, v, pos, _mask = net.encode(ids[:, :-1], start=5)
    fn = loss_fn(net, {"teacher_kl": 0.24, "evidence_weight": 8.0}, differentiate=True)
    vals, grad = nn.value_and_grad(net.step, fn)(
        net.step, pre, h, k, v, pos, ids[0, 6:], [(0, [[1, 3]])]
    )
    assert all(mx.isfinite(value).item() for value in vals)
    assert abs(vals[1].item() - vals[2].item()) < 1e-4
    assert (
        vals[4].item() > 0 and mx.linalg.norm(grad["query"]["up"]["weight"]).item() > 0
    )


def test_intermediate_prefill_writes_identical_states_without_building_output_queries(
    monkeypatch,
):
    from loopqwen import shared_query
    from loopqwen.evaluation.generation_eval import advance_cache

    net = model()
    net.step.bias = mx.full(net.step.bias.shape, 0.22)
    ids = mx.array([[3, 8, 13, 2, 34, 9, 17, 11, 28]])
    full = net.make_cache()
    net.hidden(ids[:, :6], full)
    expected = net.last_hidden(ids[:, 6:], full)

    stored = net.make_cache()
    with monkeypatch.context() as patch:

        def unused(*args, **kwargs):
            raise AssertionError("An intermediate prefix requested an output read")

        patch.setattr(shared_query, "attend", unused)
        advance_cache(net, ids[:, :6], stored)
    actual = net.last_hidden(ids[:, 6:], stored)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    for key, slot in stored.slots.items():
        actual_state = dict(tree_flatten(slot.state))
        expected_state = dict(tree_flatten(full.slots[key].state))
        assert actual_state.keys() == expected_state.keys()
        for name, a in actual_state.items():
            b = expected_state[name]
            if isinstance(a, mx.array):
                assert mx.array_equal(a, b).item()
            else:
                assert a == b


def test_shared_query_resume_restores_exact_exposure_and_optimizer(
    tmp_path, monkeypatch
):
    import pytest

    from loopqwen.io import read, save
    from loopqwen.research import shared_query_fit as fit
    from loopqwen.research.shared_query_features import arrays

    monkeypatch.setattr(fit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(fit, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fit, "CHECKPOINT_INTERVAL", 1)
    save(fit.RESULTS / fit.FEATURES / "checksums.json", {"input": "test fixture"})
    root = tmp_path / "uninterrupted"
    stop = tmp_path / "interrupted"
    evidence = tmp_path / "evidence.json"
    save(evidence, {})
    spec = {
        "id": "tiny-query",
        "lr": 2e-4,
        "fit_seed": 11,
        "epochs": 3,
        "micro_positions": 8,
        "update_positions": 8,
        "teacher_kl": 0.24,
        "evidence_weight": 8.0,
        "evidence": str(evidence),
    }
    for folder in (root, stop):
        save(folder / "registration.json", spec)
        save(folder / "environment.json", {"execution": 1})
    net = model()
    row = {"id": "one", "file": "one.safetensors", "positions": 3, "opening": 4}
    mx.save_safetensors(
        str(tmp_path / row["file"]), arrays(net, [3, 8, 13, 2], [34, 9, 17])
    )

    class Guard:
        def __init__(self, permit=None):
            self.permit = permit

        def check(self):
            if self.permit == 0:
                raise InterruptedError("Test interruption")
            if self.permit is not None:
                self.permit -= 1

    reference = fit.optimize(net, spec, [row], tmp_path, {}, root, Guard())
    monkeypatch.setattr(fit, "CACHE", tmp_path / "interrupted-cache")
    with pytest.raises(InterruptedError, match="Test interruption"):
        fit.optimize(model(), spec, [row], tmp_path, {}, stop, Guard(1))
    assert (
        read(read(stop / "checkpoint.json")["directory"] + "/state.json")["step"] == 1
    )
    actual = fit.optimize(model(), spec, [row], tmp_path, {}, stop, Guard())
    assert actual == reference
    for name in ("step.safetensors", "optimizer.safetensors"):
        a, b = (mx.load(str(path / name)) for path in (root, stop))
        assert a.keys() == b.keys() and all(
            mx.array_equal(a[k], b[k]).item() for k in a
        )
