"""Own one decoder at a time and retain exact BF16 input rows needed by a long request."""

from pathlib import Path

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from .mixed_register import build_view


class InputRows(nn.Module):
    def __init__(self, embedding, path, ids):
        super().__init__()
        path = Path(path)
        size, width = embedding.weight.shape
        if embedding.weight.dtype != mx.bfloat16:
            raise ValueError("Expected original BF16 input embeddings")
        # The sliced safetensors handle owns the mapping to this specific file. Only
        # a previously unseen generated token causes another row to be read.
        with safe_open(path, framework="mlx") as f:
            source = f.get_slice("model.embed_tokens.weight")
            if source.get_shape() != [size, width] or source.get_dtype() != "BF16":
                raise ValueError("Source embedding does not match the loaded model")
            object.__setattr__(self, "_source", source)
        identifiers = sorted(set(ids))
        if not identifiers or identifiers[0] < 0 or identifiers[-1] >= size:
            raise ValueError("Prompt token outside the original vocabulary")
        ix = mx.array(identifiers)
        self.rows = embedding(ix)
        self.index = mx.full((size,), -1, dtype=mx.int32)
        self.index[ix] = mx.arange(len(identifiers))
        mx.eval(self.rows, self.index)
        object.__setattr__(self, "_known", set(identifiers))
        object.__setattr__(self, "initial_rows", len(identifiers))

    def __call__(self, ids):
        indices = self.index[ids]
        if indices.min().item() < 0:
            missing = sorted(set(ids.reshape(-1).tolist()) - self._known)
            if not missing or missing[0] < 0 or missing[-1] >= self.index.size:
                raise ValueError("Input token outside the original vocabulary")
            n = self.rows.shape[0]
            self.rows = mx.concatenate(
                [self.rows, *[self._source[k : k + 1] for k in missing]]
            )
            self.index[mx.array(missing)] = mx.arange(n, n + len(missing))
            mx.eval(self.rows, self.index)
            self._known.update(missing)
            indices = self.index[ids]
        return self.rows[indices]


class RegisterSession:
    def __init__(self, original, reload_model, source):
        self._resident = original
        self._reload = reload_model
        self.source = source
        self.args = original.args

    def resident(self):
        if self._resident is None:
            self._resident = self._reload()
        return self._resident

    def __call__(self, *args, **kwargs):
        return self.resident()(*args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.resident(), key)

    @property
    def context_policy(self):
        return self

    @property
    def ordinary(self):
        return self.resident().context_policy.ordinary

    def views(self):
        return self.resident().context_policy.views()

    def choose(self, model, ids, cap):
        original = self.resident()
        selected, route = original.context_policy.choose(original, ids, cap)
        if selected is not original or route != "register" or len(ids) < 32768:
            return selected, route
        view, receipt = build_view(original, reset_peak=False)
        object.__setattr__(view, "fused_packed_decode", True)
        view.text.model.embed_tokens = InputRows(
            original.text.model.embed_tokens, self.source, ids
        )
        object.__setattr__(view, "weight_precision_receipt", receipt)
        self._resident = None
        return view, route
