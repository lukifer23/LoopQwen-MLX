"""Iteratively query the final Qwen attention memory without rewriting prefix history."""

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from .conditional_reentry import Correction
from .looped_model import LoopedModel
from .recurrence import Span


class QueryStep(nn.Module):
    """Transform a physical block update, preserving the input exactly at initialization."""

    def __init__(self, width, rank, gate_rank):
        super().__init__()
        self.query = Correction(width, rank)
        self.content = Correction(width, rank)
        self.gate = Correction(width, gate_rank)
        self.bias = mx.zeros((width,), dtype=mx.float32)

    def __call__(self, previous, update):
        h = previous.astype(mx.float32)
        d = update.astype(mx.float32)
        scale = mx.sqrt(mx.mean(h * h, axis=-1, keepdims=True) + 1e-6)
        gain = mx.tanh(self.bias + self.gate(h / scale))
        return (h + gain * d + scale * self.content(d / scale)).astype(previous.dtype)


def memory(layer, hidden, slot, window_size, *, mask_needed=True):
    """Append one key/value per physical token using the stock projections and RoPE."""
    a = layer.self_attn
    x = layer.input_layernorm(hidden)
    shape = (x.shape[0], x.shape[1], a.num_key_value_heads, a.head_dim)
    k = a.k_norm(a.k_proj(x).reshape(shape)).transpose(0, 2, 1, 3)
    v = a.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
    mask = (
        create_attention_mask(hidden, slot, window_size=window_size)
        if mask_needed
        else None
    )
    offset = slot.offset if slot is not None else 0
    k = a.rope(k, offset=offset)
    if slot is not None:
        k, v = slot.update_and_fetch(k, v)
    return k, v, offset, mask


def differentiable_attention(q, k, v, scale, mask):
    """Grouped-query attention with an explicit MLX backward path for short suffixes."""
    batch, heads, length, dim = q.shape
    kv_heads, context = k.shape[1:3]
    grouped = q.reshape(batch, kv_heads, heads // kv_heads, length, dim)
    scores = (grouped.astype(mx.float32) * scale) @ k.astype(mx.float32)[
        :, :, None
    ].swapaxes(-2, -1)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("Unsupported shared-query attention mask")
        mask = (
            mx.arange(context)[None, :] <= mx.arange(context - length, context)[:, None]
        )
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[:, :, None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    output = weights @ v.astype(mx.float32)[:, :, None]
    return output.reshape(batch, heads, length, dim).astype(q.dtype)


def attention_update(
    layer,
    hidden,
    k,
    v,
    offset,
    mask,
    *,
    step=None,
    observe_query=None,
    differentiate=False,
):
    """Read the projected causal history without appending keys or values."""
    a = layer.self_attn
    batch, length, _ = hidden.shape
    source = layer.input_layernorm(hidden)
    if step is not None:
        source = (source + step.query(source.astype(mx.float32))).astype(source.dtype)
    projection = a.q_proj(source)
    q, gate = mx.split(
        projection.reshape(batch, length, a.num_attention_heads, -1), 2, axis=-1
    )
    q = a.rope(a.q_norm(q).transpose(0, 2, 1, 3), offset=offset)
    if observe_query is not None:
        observe_query(q)
    if differentiate:
        kernel = differentiate if callable(differentiate) else differentiable_attention
        z = kernel(q, k, v, a.scale, mask)
    else:
        z = scaled_dot_product_attention(q, k, v, cache=None, scale=a.scale, mask=mask)
    z = z.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return a.o_proj(z * mx.sigmoid(gate.reshape(batch, length, -1)))


def attend(
    layer,
    hidden,
    k,
    v,
    offset,
    mask,
    *,
    step=None,
    observe_query=None,
    differentiate=False,
):
    """Run the original gated query, output projection and MLP on selected positions."""
    h = hidden + attention_update(
        layer,
        hidden,
        k,
        v,
        offset,
        mask,
        step=step,
        observe_query=observe_query,
        differentiate=differentiate,
    )
    return h + layer.mlp(layer.post_attention_layernorm(h))


class SharedQuery(LoopedModel):
    """One write for each input at each ordinary site, with tied read-only output visits."""

    def __init__(self, text, *, revisits, rank, gate_rank):
        if revisits not in (0, 2) or text.layers[-1].is_linear:
            raise ValueError(
                "Shared queries require a final attention block and zero or two revisits"
            )
        # Sites enumerate state owners. Read-only visits do not own another temporal cache.
        super().__init__(text, Span(start=len(text.layers) - 1, length=1, passes=1))
        self.freeze()
        self.revisits = revisits
        self.output_positions = 0
        self.output_revisits = 0
        self.step = QueryStep(text.args.hidden_size, rank, gate_rank)

    def _write_memory(self, ids, cache, window_size, observer, *, mask_needed):
        if cache is not None:
            cache.validate(self, window_size)
        h = self._execute(
            self.text.model.embed_tokens(ids),
            self.sites[:-1],
            cache,
            window_size,
            observer,
        )
        slot = cache.slots[self.sites[-1]] if cache is not None else None
        layer = self.text.layers[-1]
        return h, *memory(layer, h, slot, window_size, mask_needed=mask_needed)

    def populate_cache(self, ids, cache):
        """Update causal slots without constructing any final output reads or MLP."""
        self._write_memory(ids, cache, None, None, mask_needed=False)

    def encode(self, ids, cache=None, *, start=0, window_size=None, observer=None):
        h, k, v, offset, mask = self._write_memory(
            ids, cache, window_size, observer, mask_needed=True
        )
        layer = self.text.layers[-1]
        if not 0 <= start < h.shape[1]:
            raise ValueError(
                "The first required output must fall inside the supplied token block"
            )
        h = h[:, start:]
        if isinstance(mask, mx.array):
            mask = mask[..., start:, :]
        elif h.shape[1] == 1:
            mask = None
        pos = offset + start
        ordinary = attend(layer, h, k, v, pos, mask)
        if observer is not None:
            observer("exit", 0, ordinary)
        return h, ordinary, k, v, pos, mask

    def refine(
        self,
        pre,
        ordinary,
        k,
        v,
        pos,
        mask,
        *,
        step=None,
        observer=None,
        differentiate=False,
        observe_query=None,
        visits=None,
    ):
        visits = self.revisits if visits is None else visits
        if visits not in ((1, 2) if self.revisits else (0,)):
            raise ValueError("Invalid physical visit intervention")
        step = self.step if step is None else step
        layer = self.text.layers[-1]
        if not self.revisits:
            raw = attend(
                layer,
                pre,
                k,
                v,
                pos,
                mask,
                step=step,
                observe_query=observe_query,
                differentiate=differentiate,
            )
            return step(raw, raw - pre)
        state = ordinary
        for visit in range(visits):
            proposal = attend(
                layer,
                state,
                k,
                v,
                pos,
                mask,
                step=step,
                observe_query=observe_query if visit == visits - 1 else None,
                differentiate=differentiate,
            )
            state = step(state, proposal - state)
            if observer is not None:
                observer("exit", visit + 1, state)
        return state

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        features = self.encode(ids, cache, window_size=window_size, observer=observer)
        return self.text.model.norm(self.refine(*features, observer=observer))

    def _final_output(self, features):
        self.output_positions += features[0].shape[0]

        def count_visit(_, visit, state):
            self.output_revisits += state.shape[0]

        return self.refine(*features, observer=count_visit)

    def staged_final(self, hidden, slot, *, emit):
        """Write the last site's full memory and run the learned reads only at an output."""
        layer = self.text.layers[-1]
        keys, values, offset, _ = memory(layer, hidden, slot, None, mask_needed=False)
        if not emit:
            mx.eval(keys, values)
            return None
        position = offset + hidden.shape[1] - 1
        pre = hidden[:, -1:]
        ordinary = attend(layer, pre, keys, values, position, None)
        return self._final_output((pre, ordinary, keys, values, position, None))

    def last_hidden(self, ids, cache):
        features = self.encode(ids, cache, start=ids.shape[1] - 1)
        return self.text.model.norm(self._final_output(features))
