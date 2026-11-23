import math

import mlx.core as mx

from ..data.tokenize import batch


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


def validation_nll(model, samples, batch_size=4, projection_chunk=None):
    model.eval()
    total = labels = 0
    ordered = sorted(samples, key=lambda r: r["input_length"])
    for i in range(0, len(ordered), batch_size):
        x, p, y = batch(ordered[i : i + batch_size])
        if projection_chunk is None:
            total += loss_sum(model, x, p, y).item()
        else:
            h = model.hidden(x)
            selected = h.reshape(-1, h.shape[-1])[p]
            mx.eval(selected)
            for j in range(0, y.size, projection_chunk):
                logits = model.project(selected[j : j + projection_chunk])
                total += (
                    mx.fast.cross_entropy(logits, y[j : j + projection_chunk])
                    .sum()
                    .item()
                )
        labels += y.size
    return dict(nll=total / labels, perplexity=math.exp(total / labels), tokens=labels)
