import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from test_shared_query import model
from loopqwen.research.shared_query_features import arrays
from loopqwen.source_copy import SourceCopyRead
from scripts.receipt_repair_fit import probabilities, targets, terms, streamed_arrays


def test_report_contrast_uses_only_observed_tokens_and_trains_both_paths():
    base = model()
    net = SourceCopyRead(base.text, revisits=2, rank=8, gate_rank=4)
    net.eval()
    row = dict(
        group="repair", p=[3, 4, 5], y=[6, 7, 8, 9, 10], negatives=[[1, 12], [3, 13]]
    )
    data = arrays(net, row["p"], row["y"])
    segmented = streamed_arrays(net, row, block=1)
    for field in ("pre", "ordinary", "keys", "values"):
        assert mx.allclose(segmented[field], data[field], atol=2e-4).item()
    data = segmented
    data.update(targets(row))
    whole = net(mx.array([row["p"] + row["y"][:-1]]))[:, 2:]
    labels = data["labels"]
    expected = mx.take_along_axis(whole, labels[..., None], axis=-1)[..., 0]
    for start, end in ((0, 2), (2, 3), (3, 8)):
        y, margins = probabilities(net, net.step, data, start, end, 2, 2, False)
        assert mx.allclose(y, expected[:, start:end], atol=2e-4).item()
        for j, (pos, alt) in enumerate(row["negatives"]):
            if start <= pos < end:
                diff = expected[:, pos] - whole[:, pos, alt]
                assert mx.allclose(margins[:, pos - start, j], diff, atol=2e-4).item()

    def objective(step):
        return terms(net, step, data, row, 0, 2, 2, True)[0]

    _, g = nn.value_and_grad(net.step, objective)(net.step)
    gradients = dict(tree_flatten(g))
    for key in ("pointer.gate.weight", "write.weight"):
        assert (
            mx.isfinite(gradients[key]).all() & (mx.abs(gradients[key]).sum() > 0)
        ).item()
