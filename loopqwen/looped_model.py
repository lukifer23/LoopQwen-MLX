import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .recurrence import Span


class LoopedModel(nn.Module):
    def __init__(self, text, span=Span(), alpha=0.1):
        super().__init__()
        self._generation = object()
        self.text = text
        self.span = span
        span.sites(len(text.layers))
        self.checkpointing = False
        self.alpha_override = None
        if span.has_gate:
            if not 0 < alpha < 1:
                raise ValueError(
                    "Initial interpolation must be strictly between 0 and 1"
                )
            self.alpha_logit = mx.array(math.log(alpha / (1 - alpha)))

    @property
    def args(self):
        return self.text.args

    @property
    def sites(self):
        return self.span.sites(len(self.text.layers))

    def make_cache(self, max_kv_size=None, *, left_padding=None):
        from .recurrent_cache import SiteCache

        return SiteCache(self, max_kv_size, left_padding=left_padding)

    def effective_alpha(self):
        return (
            self.alpha_override
            if self.alpha_override is not None
            else mx.sigmoid(self.alpha_logit)
        )

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        h = self.text.model.embed_tokens(ids)
        sites = self.sites
        if cache is not None:
            cache.validate(self, window_size)
        return self.text.model.norm(
            self._execute(h, sites, cache, window_size, observer)
        )

    def update(self, parameters, strict=True):
        result = super().update(parameters, strict=strict)
        self._generation = object()
        return result

    def block_states(self, ids, passes, refinement_observer=None):
        """Uncached training exits. Expose no target tokens or states on re-entry."""
        start = self.span.start
        end = start + self.span.length
        h = self._execute(
            self.text.model.embed_tokens(ids), [(i, 0) for i in range(start)]
        )
        anchor = h
        for p in range(passes):
            previous = h
            if p and self.span.mode == "input_anchor":
                h = (h + self.effective_alpha() * (anchor - h)).astype(h.dtype)
            h = self._execute(h, [(i, 0) for i in range(start, end)])
            candidate = h
            if p and self.span.mode == "stabilized":
                h = (previous + self.effective_alpha() * (candidate - previous)).astype(
                    h.dtype
                )
            if p and refinement_observer is not None:
                refinement_observer(previous, candidate, h)
            yield h

    def suffix(self, h):
        end = self.span.start + self.span.length
        return self.text.model.norm(
            self._execute(h, [(i, 0) for i in range(end, len(self.text.layers))])
        )

    def _execute(self, h, sites, cache=None, window_size=None, observer=None):
        for i, p in sites:
            layer = self.text.layers[i]
            state = None if cache is None else cache.slots[i, p]
            mask = (
                create_ssm_mask(h, state)
                if layer.is_linear
                else create_attention_mask(h, state, window_size=window_size)
            )
            if self.span.mode == "input_anchor" and i == self.span.start:
                if p == 0:
                    anchor = h
                else:
                    alpha = self.effective_alpha()
                    h = (h + alpha * (anchor - h)).astype(h.dtype)
            if self.span.mode == "stabilized" and p and i == self.span.start:
                previous = h
            if observer is not None and i == self.span.start:
                observer("entry", p, h)
            if (
                self.training
                and self.checkpointing
                and (
                    getattr(self, "checkpoint_policy", "all") == "all"
                    or self.span.start <= i < self.span.start + self.span.length
                )
            ):
                # Capture masks in the callable; only tensors are checkpoint inputs.
                h = nn.utils.checkpoint(layer, lambda x, l=layer, m=mask: l(x, mask=m))(
                    h
                )
            else:
                h = layer(h, mask=mask, cache=state)
            if (
                self.span.mode == "stabilized"
                and p
                and i == self.span.start + self.span.length - 1
            ):
                alpha = (
                    self.alpha_override
                    if self.alpha_override is not None
                    else mx.sigmoid(self.alpha_logit)
                )
                h = (previous + alpha * (h - previous)).astype(h.dtype)
            if observer is not None and i == self.span.start + self.span.length - 1:
                observer("exit", p, h)
        return h

    def project(self, hidden):
        if self.args.tie_word_embeddings:
            return self.text.model.embed_tokens.as_linear(hidden)
        return self.text.lm_head(hidden)

    def __call__(self, ids, cache=None):
        return self.project(self.hidden(ids, cache))
