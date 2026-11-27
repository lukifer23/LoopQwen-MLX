"""A learned pointer–generator over original causal token IDs and contextual values."""

import mlx.core as mx
from mlx import nn

from .bounded_read import BoundedContentRead
from .conditional_reentry import Correction


class SourceMixture(nn.Module):
    def __init__(self, width, heads, dimension, rank):
        super().__init__()
        self.heads = heads
        self.dimension = dimension
        self.address = nn.Linear(width, heads * dimension, bias=False)
        self.gate = nn.Linear(width, 1)
        self.gate.weight = mx.zeros_like(self.gate.weight)
        self.gate.bias = mx.array([-4.6])
        self.language = Correction(width, rank)

    def distributions(self, hidden, values, position, project):
        """Compute causal address probabilities and a normalized vocabulary distribution."""
        x = hidden.astype(mx.float32)
        norm = mx.fast.rms_norm(x, None, 1e-6)
        query = self.address(norm).reshape(*x.shape[:2], self.heads, self.dimension)
        b, length = x.shape[:2]
        if position < 0 or position + length > values.shape[2]:
            raise ValueError("Pointer decisions must lie inside the observed source")
        q = query.transpose(0, 2, 1, 3) * self.dimension**-0.5
        # Cast the small score array for softmax. The entire immutable KV remains
        # in native precision, including for a single decision on a long document.
        score = (q.astype(values.dtype) @ values.swapaxes(-1, -2)).astype(mx.float32)
        allowed = (
            mx.arange(values.shape[2])[None, :]
            <= (position + mx.arange(length))[:, None]
        )
        copy = mx.softmax(mx.where(allowed, score, -mx.inf), axis=-1).mean(axis=1)
        corrected = x + 0.25 * mx.tanh(self.language(norm))
        logits = project(corrected.astype(hidden.dtype)).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        gate = self.gate(norm).reshape(b, length)
        return logp, copy, gate

    @staticmethod
    def combine(vocabulary, source, gate):
        # log-sigmoid remains stable at both ends, including unseen token IDs.
        source_log = mx.where(source > 0, mx.log(mx.maximum(source, 1e-30)), -mx.inf)
        return mx.logaddexp(
            vocabulary - nn.softplus(gate), source_log - nn.softplus(-gate)
        )

    def selected(self, hidden, values, ids, position, labels, project):
        logp, attention, gate = self.distributions(hidden, values, position, project)
        selected = mx.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
        mass = (attention * (ids[:, None, :] == labels[:, :, None])).sum(axis=-1)
        return self.combine(selected, mass, gate), mx.sigmoid(gate), mass

    def full(self, hidden, values, ids, position, project):
        logp, attention, gate = self.distributions(hidden, values, position, project)
        batch, length = hidden.shape[:2]
        if ids.shape != (batch, values.shape[2]):
            raise ValueError(
                "Token IDs must correspond one-to-one to the physical source values"
            )
        mass = (
            mx.zeros_like(logp)
            .at[
                mx.arange(batch)[:, None, None],
                mx.arange(length)[None, :, None],
                ids[:, None, :],
            ]
            .add(attention)
        )
        return self.combine(logp, mass, gate[..., None])


class SourceCopyRead(BoundedContentRead):
    def __init__(self, text, *, revisits, rank, gate_rank):
        super().__init__(text, revisits=revisits, rank=rank, gate_rank=gate_rank)
        self.step.pointer = SourceMixture(
            text.args.hidden_size, self.step.heads, self.step.dimension, rank
        )

    @staticmethod
    def remember(cache, ids):
        if cache is None:
            return
        previous = getattr(cache, "source_ids", None)
        cache.source_ids = (
            ids if previous is None else mx.concatenate([previous, ids], axis=1)
        )

    def _write_memory(self, ids, cache, window_size, observer, *, mask_needed):
        result = super()._write_memory(
            ids, cache, window_size, observer, mask_needed=mask_needed
        )
        self.remember(cache, ids)
        return result

    def project_cached(self, hidden, cache):
        slot = cache.slots[self.sites[-1]]
        if hasattr(slot, "source_values"):
            values = slot.source_values()
        else:
            _, values = slot.keys_and_values()
        return self.step.pointer.full(
            hidden,
            values,
            cache.source_ids,
            values.shape[2] - hidden.shape[1],
            self.project,
        )

    def __call__(self, ids, cache=None):
        if cache is None:
            pre, h, k, v, pos, mask = self.encode(ids)
            hidden = self.text.model.norm(self.refine(pre, h, k, v, pos, mask))
            return self.step.pointer.full(hidden, v, ids, pos, self.project)
        return self.project_cached(self.hidden(ids, cache), cache)
