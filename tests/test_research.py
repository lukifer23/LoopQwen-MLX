"""Integration tests execute the actual pretrained network on Metal."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from loopqwen.benchmark.bench_gdn import bench_gdn, check_projection_gradients, relative
from loopqwen.cache_check import cache_check
from loopqwen.checkpoint import restore, setup, write
from loopqwen.config import config
from loopqwen.data.build_data import load_split
from loopqwen.data.tokenize import batch
from loopqwen.evaluation.math_eval import normalized
from loopqwen.kernels.gdn_backend import select
from loopqwen.looped_model import LoopedModel
from loopqwen.model_loader import load_base
from loopqwen.probe import probe
from loopqwen.recurrence import Span
from loopqwen.training.adapters import attach
from loopqwen.training.gradient_check import gradient_check
from loopqwen.training.losses import loss_sum
from loopqwen.training.optimizers import optimizer


def test_pretrained_and_schedule():
    r = probe(Span())
    assert r["unique"] == 752393024 == r["wrapped_unique"]
    assert r["k1_max_abs"] == 0
    assert [a["executions"] for a in r["rows"]] == [24, 30]
    with pytest.raises(ValueError):
        Span(21, 6, 2).sites(24)


def test_cached_full_parity():
    rows = cache_check()["rows"]
    assert max(r["length"] for r in rows) >= 32
    assert {r["chunk"] for r in rows} == {1, 7}


def test_real_gdn():
    from loopqwen.benchmark.bench_gdn import masked_contract

    masked_contract()
    bench_gdn()
    check_projection_gradients()


def test_optimizer_and_gradient():
    state = gradient_check()
    assert state["rows"][1]["min_gradient"] > 0


@pytest.mark.parametrize("backend,tolerance", [("reference", 1e-5), ("native", 0.025)])
def test_each_call_contributes_to_the_shared_gradient(backend, tolerance):
    model = LoopedModel(load_base()[0])
    mx.random.seed(41)
    attach(model, projections=["up_proj"])
    select(model, backend)
    model.train()
    adapter = model.text.layers[12].mlp.up_proj
    # A test-only switch permits differentiating one use of the shared weight
    # at a time. In every case the full hidden-state derivative passes through.
    calls = []
    allowed = {0, 1}

    class Counted(LoRALinear):
        def __call__(self, x):
            call = len(calls)
            calls.append(call)
            w = self.lora_b if call in allowed else mx.stop_gradient(self.lora_b)
            return self.linear(x) + (self.scale * ((x @ self.lora_a) @ w)).astype(
                x.dtype
            )

    object.__setattr__(adapter, "__class__", Counted)
    x, pos, y = batch(load_split("train")[:1])
    fn = nn.value_and_grad(model, loss_sum)
    gradients = []
    for enabled in ({0, 1}, {0}, {1}):
        allowed = enabled
        calls.clear()
        _, g = fn(model, x, pos, y)
        actual = dict(tree_flatten(g))["text.model.layers.12.mlp.up_proj.lora_b"]
        mx.eval(actual)
        assert calls == [0, 1] and mx.linalg.norm(actual).item() > 0
        gradients.append(actual)
    assert relative(gradients[0], gradients[1] + gradients[2]) < tolerance


def test_labels_splits_and_normalization():
    splits = [load_split(name) for name in ("train", "validation", "test")]
    ids = [r["id"] for rows in splits for r in rows]
    assert len(ids) == len(set(ids))
    rows = splits[0][:4]
    x, pos, y = batch(rows)
    assert y.tolist() == [token for r in rows for token in r["ids"][r["prefix"] :]]
    assert x.flatten()[pos].tolist() == [
        token for r in rows for token in r["ids"][r["prefix"] - 1 : -1]
    ]
    assert normalized(" 2/4\nQuestion:") == normalized("0.50")
    assert normalized("1 or 2") is None


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_checkpoint_resume_and_reproducibility(tmp_path, backend):
    c = config("configs/smoke.yaml")
    c["gdn_backend"] = backend
    x, pos, y = batch(load_split("train")[:1])

    def backward(model):
        model.train()
        value, grads = nn.value_and_grad(model, loss_sum)(model, x, pos, y)
        mx.eval(value, grads)
        return value.item(), grads

    def apply(model, opt, grads):
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)

    def snapshot(model, opt):
        return dict(tree_flatten(model.trainable_parameters())), dict(
            tree_flatten(opt.state)
        )

    def assert_same(live, saved):
        for a, b in zip(live, saved, strict=True):
            assert a.keys() == b.keys()
            assert all(mx.array_equal(a[k], v).item() for k, v in b.items())

    model, _, _ = setup(c)
    opt = optimizer(c)
    initial, first_grad = backward(model)
    apply(model, opt, first_grad)
    model.eval()
    saved = model(x)
    mx.eval(saved)
    at_save = snapshot(model, opt)
    write(tmp_path, model, opt, dict(config=c, step=1))
    continued, continuation_grad = backward(model)
    apply(model, opt, continuation_grad)
    expected = snapshot(model, opt)
    model.eval()
    expected_logits = model(x).reshape(-1, model.args.vocab_size)[pos]
    mx.eval(expected_logits)
    del model, opt

    model, _, _ = setup(c)
    opt = optimizer(c)
    repeated, grad = backward(model)
    assert repeated == initial
    apply(model, opt, grad)
    restore(tmp_path, model, opt)
    assert_same(snapshot(model, opt), at_save)
    model.eval()
    assert mx.array_equal(saved, model(x)).item()

    resumed, independent_grad = backward(model)
    assert resumed == continued
    apply(model, opt, independent_grad)
    if backend == "reference":
        assert_same(snapshot(model, opt), expected)
    else:
        # A separately executed native backward uses atomic reductions. Compare
        # its effect on all scored logits under the established BF16 tolerance.
        model.eval()
        actual = model(x).reshape(-1, model.args.vocab_size)[pos]
        assert relative(actual, expected_logits) < 0.03
        assert mx.array_equal(
            mx.argmax(actual, -1), mx.argmax(expected_logits, -1)
        ).item()

    # With the exact same real backward result, every resumed AdamW moment,
    # schedule tensor and adapter must equal its uninterrupted counterpart.
    restore(tmp_path, model, opt)
    apply(model, opt, continuation_grad)
    assert_same(snapshot(model, opt), expected)


def test_stabilized_gate_trains():
    c = config("configs/stabilized_loop.yaml")
    model, _, _ = setup(c)
    model.train()
    x, pos, y = batch(load_split("validation")[:1])
    _, g = nn.value_and_grad(model, loss_sum)(model, x, pos, y)
    assert 0 < abs(g["alpha_logit"].item()) < 1e5
    old = model.alpha_logit
    optimizer(c).update(model, g)
    assert model.alpha_logit.item() != old.item()


def test_packing_resets_state_at_boundaries():
    from loopqwen.data.tokenize import pack
    from loopqwen.training.losses import packed_loss_sum

    model, _, _ = setup(config("configs/smoke.yaml"))
    model.train()
    rows = load_split("train")[:2]
    val, grad = nn.value_and_grad(model, packed_loss_sum)(model, *pack(rows))

    def separate(m):
        return sum(loss_sum(m, *batch([r])) for r in rows)

    baseline, ref = nn.value_and_grad(model, separate)(model)
    assert (
        abs((val - baseline).item()) / sum(len(r["ids"]) - r["prefix"] for r in rows)
        < 0.01
    )
    assert (
        max(
            relative(a, b)
            for (_, a), (_, b) in zip(tree_flatten(grad), tree_flatten(ref))
        )
        < 0.025
    )
