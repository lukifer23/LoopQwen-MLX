import mlx.core as mx


def loss_sum(model, x, positions, labels):
    # Select only supervised hidden rows before the expensive tied vocabulary projection.
    h = model.hidden(x)
    selected = h.reshape(-1, h.shape[-1])[positions]
    return mx.fast.cross_entropy(model.project(selected), labels).sum()
