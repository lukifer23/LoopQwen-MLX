"""Learned full-history attention bias. Input consists solely of model token IDs."""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import KVCache


def normalized(x):
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-7)


class PassageScorer(nn.Module):
    def __init__(self, hidden, width=32, query_length=64, span_width=12, stride=4):
        super().__init__()
        self.projection = nn.Linear(hidden, width, bias=False)
        self.salience = nn.Linear(width * 3, 1)
        self.position = mx.zeros((query_length,))
        self.query_length = query_length
        self.span_width = span_width
        self.stride = stride

    def features(self, embeddings):
        x = normalized(self.projection(embeddings.astype(mx.float32)))
        z = mx.zeros_like(x[:1])
        return normalized(
            mx.concatenate(
                [
                    0.7 * mx.concatenate([z, x[:-1]]),
                    x,
                    0.7 * mx.concatenate([x[1:], z]),
                ],
                axis=-1,
            )
        )

    def pieces(self, source, query, importance):
        # Overlapping windows in blocks. Retain every candidate even at 131K.
        total = (source.shape[0] - self.span_width) // self.stride + 1
        blocks = []
        for start in range(0, total, 256):
            count = min(256, total - start)
            offset = start * self.stride
            k = source[offset : offset + (count - 1) * self.stride + self.span_width]
            sim = query @ k.T
            windows = (
                mx.arange(count)[:, None] * self.stride
                + mx.arange(self.span_width)[None, :]
            )
            blocks.append(importance @ mx.max(sim[:, windows], axis=-1))
        return mx.concatenate(blocks)

    def __call__(self, embeddings):
        # The source and query never share a token position.
        src = self.features(embeddings[: -self.query_length])
        query = self.features(embeddings[-self.query_length :])
        weights = mx.softmax(self.salience(query).squeeze(-1) + self.position)
        return self.pieces(src, query, weights)

    def infer(self, embedding, ids):
        # Project bounded blocks of the large embedding table gather. A one-token
        # overlap at each end reproduces the two neighboring features exactly.
        n = len(ids) - self.query_length
        q = self.features(embedding(mx.array(ids[n:])))
        importance = mx.softmax(self.salience(q).squeeze(-1) + self.position)
        blocks = []
        for start in range(0, n, 2048):
            stop = min(start + 2048, n)
            a = max(0, start - 1)
            v = self.features(embedding(mx.array(ids[a : min(stop + 1, n)])))[
                start - a : stop - a
            ]
            mx.eval(v)
            blocks.append(v)
        scores = self.pieces(mx.concatenate(blocks), q, importance)
        return mx.argmax(scores).item() * self.stride


class AttentionReader:
    def __init__(self, scorer, gain, minimum):
        self.scorer = scorer
        self.gain = gain
        self.minimum = minimum

    def select(self, model, ids):
        if len(ids) < self.minimum:
            return None
        start = self.scorer.infer(model.text.model.embed_tokens, ids)
        return {
            "start": max(0, start - 2),
            "stop": min(
                len(ids) - self.scorer.query_length, start + self.scorer.span_width + 12
            ),
            "gain": self.gain + max(0.0, math.log(len(ids) / 2048)),
            "length": len(ids),
        }


def attach_bias(cache, section):
    """Install once per attention site, retaining the site's exact offset and KV tensors."""
    for (layer, _), slot in cache.slots.items():
        if (
            layer < 8
            or not isinstance(slot, KVCache)
            or getattr(slot, "reader_bias", False)
        ):
            continue
        original = slot.make_mask

        def mask(
            n, *, return_array=False, window_size=None, slot=slot, original=original
        ):
            end = slot.offset + n
            if end < section["length"]:
                return original(n, return_array=return_array, window_size=window_size)
            keys = mx.arange(end)
            boost = ((keys >= section["start"]) & (keys < section["stop"])).astype(
                mx.float32
            ) * section["gain"]
            if n == 1:
                return boost[None, None, None, :]
            queries = mx.arange(slot.offset, end)[:, None]
            causal = mx.where(queries >= keys[None, :], 0.0, -float("inf"))
            return (
                causal + mx.where(queries >= section["length"] - 1, boost[None, :], 0.0)
            )[None, None, :, :]

        slot.make_mask = mask
        slot.reader_bias = True


def load_reader(model, directory):
    root = Path(directory)
    meta = json.loads((root / "attention-reader.json").read_bytes())
    path = root / "attention-reader.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["scorer_sha256"]:
        raise ValueError("Attention reader weights differ")
    scorer = PassageScorer(model.args.hidden_size, **meta["architecture"])
    scorer.load_weights(str(path), strict=True)
    scorer.eval()
    object.__setattr__(
        model,
        "attention_reader",
        AttentionReader(scorer, meta["gain"], meta["minimum"]),
    )
