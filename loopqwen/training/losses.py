import mlx.core as mx


def loss_sum(model, x, positions, labels):
    # Select only supervised hidden rows before the expensive tied vocabulary projection.
    h = model.hidden(x)
    selected = h.reshape(-1, h.shape[-1])[positions]
    return mx.fast.cross_entropy(model.project(selected), labels).sum()


def packed_loss_sum(model, x, positions, labels, lengths):
    # Reset both hybrid decoder families at each boundary. Apply the vocabulary
    # head once across the gathered scored rows, with no cross-document targets.
    offset = 0
    segments = []
    for length in lengths:
        ids = x[None, offset : offset + length]
        if length < 16:
            ids = mx.pad(ids, [(0, 0), (0, 16 - length)])
        segments.append(model.hidden(ids)[0, :length])
        offset += length
    h = mx.concatenate(segments, 0)[positions]
    return mx.fast.cross_entropy(model.project(h), labels).sum()
