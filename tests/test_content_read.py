import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from test_shared_query import model

from loopqwen.content_read import ContentRead


def test_content_read_preserves_parent_then_uses_causal_values_at_both_visits():
    parent = model()
    net = ContentRead(parent.text, revisits=2, rank=8, gate_rank=4)
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    net.eval()
    x = mx.array([[1, 3, 5, 7, 9, 11]])
    assert mx.array_equal(net(x), parent(x)).item()
    net.step.write.weight = mx.random.normal(net.step.write.weight.shape) * 0.04
    calls = []
    pre, h, k, v, pos, mask = net.encode(x)
    z = net.refine(
        pre, h, k, v, pos, mask, observer=lambda *args: calls.append(args[1])
    )
    assert calls == [1, 2] and not mx.allclose(z, h).item()
    other = mx.array([[1, 3, 5, 7, 54, 55]])
    assert mx.allclose(net(x)[:, :4], net(other)[:, :4], rtol=1e-5, atol=1e-5).item()
    cache = net.make_cache()
    net.last_hidden(x[:, :4], cache)
    assert mx.allclose(
        net.project(net.last_hidden(x[:, 4:], cache)), net(x)[:, -1:], atol=2e-4
    ).item()

    def loss(step):
        states = net.refine(pre, h, k, v, pos, mask, step=step, differentiate=True)
        return nn.losses.cross_entropy(
            net.project(net.text.model.norm(states[:, :-1])), x[:, 1:], reduction="mean"
        )

    _, g = nn.value_and_grad(net.step, loss)(net.step)
    for k in ("write", "address"):
        norm = mx.linalg.norm(g[k]["weight"]).item()
        assert 0 < norm < float("inf")
