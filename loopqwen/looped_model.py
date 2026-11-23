import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .recurrence import Span


class LoopedModel(nn.Module):
    def __init__(self, text, span=Span(), alpha=0.1):
        super().__init__()
        self.text = text
        self.span = span
        span.sites(len(text.layers))
        self.checkpointing = False
        self.alpha_override = None
        if span.mode == "stabilized":
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

    def make_cache(self, max_kv_size=None):
        from .recurrent_cache import SiteCache

        return SiteCache(self, max_kv_size)

    def hidden(self, ids, cache=None):
        h = self.text.model.embed_tokens(ids)
        sites = self.sites
        if cache is not None and (cache.sites != sites or self.training):
            raise ValueError(
                "Cache requires evaluation mode and its original execution schedule"
            )
        for i, p in sites:
            layer = self.text.layers[i]
            state = None if cache is None else cache.slots[i, p]
            mask = (create_ssm_mask if layer.is_linear else create_attention_mask)(
                h, state
            )
            if self.span.mode == "stabilized" and p and i == self.span.start:
                previous = h
            if self.training and self.checkpointing:
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
        return self.text.model.norm(h)

    def project(self, hidden):
        if self.args.tie_word_embeddings:
            return self.text.model.embed_tokens.as_linear(hidden)
        return self.text.lm_head(hidden)

    def __call__(self, ids, cache=None):
        return self.project(self.hidden(ids, cache))
