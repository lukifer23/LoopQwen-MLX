import mlx.core as mx
from mlx import nn
from test_shared_query import model

from loopqwen.source_copy import SourceCopyRead, SourceMixture
from loopqwen.tool_prefill import ToolPrefix


def test_mixture_is_causal_normalized_and_matches_selected_token_derivative():
    mx.random.seed(2381)
    head = SourceMixture(16, 2, 8, 4)
    projection = nn.Linear(16, 19)
    h = mx.random.normal((2, 3, 16))
    v = mx.random.normal((2, 2, 5, 8))
    ids = mx.array([[3, 3, 4, 5, 8], [2, 5, 2, 2, 12]])
    labels = mx.array([[3, 18, 5], [2, 8, 12]])
    log = head.full(h, v, ids, 2, projection)
    sel, _, _ = head.selected(h, v, ids, 2, labels, projection)
    assert mx.allclose(mx.exp(log).sum(axis=-1), mx.ones((2, 3)), atol=1e-5).item()
    assert mx.allclose(
        mx.take_along_axis(log, labels[..., None], axis=-1)[..., 0], sel
    ).item()
    v2 = v.at[:, :, 4:].add(87)
    assert mx.allclose(log[:, :2], head.full(h, v2, ids, 2, projection)[:, :2]).item()

    def full(x):
        return mx.take_along_axis(
            head.full(x, v, ids, 2, projection), labels[..., None], axis=-1
        ).sum()

    def selected(x):
        return head.selected(x, v, ids, 2, labels, projection)[0].sum()

    assert mx.allclose(mx.grad(full)(h), mx.grad(selected)(h), atol=1e-5).item()


def test_cache_reuses_real_token_identities_across_request_boundaries():
    base = model()
    net = SourceCopyRead(base.text, revisits=2, rank=8, gate_rank=4)
    net.eval()
    net.step.pointer.gate.bias = mx.array([2.0])
    seq = [9, 13, 9, 22, 3, 7, 10, 11, 2]
    cache = net.make_cache()
    net.populate_cache(mx.array([seq[:4]]), cache)
    prefix = ToolPrefix(max_tokens=4)
    prefix.prepare(net, seq, 4)
    prefix.capture(net, seq, 4, cache)
    second = net.make_cache()
    assert prefix.begin(net, seq, 4, second) == 4
    out = net(mx.array([seq[4:]]), second)
    full = net(mx.array([seq]))
    assert mx.allclose(out, full[:, 4:], rtol=2e-4, atol=2e-4).item()
    assert second.source_ids.tolist() == [seq]
