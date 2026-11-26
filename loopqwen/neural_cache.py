"""Causal passage memory keyed by a second use of the final native attention layer."""

import mlx.core as mx

from .looped_model import LoopedModel
from .recurrence import Span
from .tool_model import quantize_prefix


def normalize(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-8)


def weights(parameters, similarity, allowed):
    temperature = mx.exp(mx.clip(parameters[0], 0, 4.5))
    scores = mx.where(allowed, similarity, -1e4)
    total = mx.softmax(scores * temperature, axis=-1) * allowed
    best = mx.max(scores, axis=-1)
    gate = mx.sigmoid(parameters[1] + parameters[2] * (best - 0.5)) * mx.any(
        allowed, axis=-1
    )
    return total, gate


class RecurrentMemory:
    def __init__(self, text, parameters, bits=8, passes=2):
        quantize_prefix(text, 23, bits)
        self.decoder = LoopedModel(
            text, Span(start=23, length=1, passes=passes, mode="stabilized"), alpha=0.35
        )
        self.decoder.eval()
        self.parameters = mx.array(parameters)
        self.passes = passes

    def representations(self, ids, cache=None):
        net = self.decoder
        hidden = net.text.model.embed_tokens(ids)
        hidden = net._execute(hidden, net.sites[:24], cache)
        ordinary = net.text.model.norm(hidden)
        if self.passes == 2:
            hidden = net._execute(hidden, net.sites[24:], cache)
        return ordinary[0], normalize(net.text.model.norm(hidden)[0])

    def correct(self, probs, queries, keys, values, allowed):
        if not keys.shape[0]:
            return probs
        distribution, gate = weights(self.parameters, queries @ keys.T, allowed)
        out = probs * (1 - gate[:, None])
        return out.at[mx.arange(queries.shape[0])[:, None], values[None, :]].add(
            gate[:, None] * distribution
        )

    def generate(self, prompt_ids, max_tokens, stop_ids=(), memory_length=256):
        if (
            not prompt_ids
            or max_tokens < 1
            or memory_length < 1
            or len(prompt_ids) + max_tokens > self.decoder.args.max_position_embeddings
        ):
            raise ValueError("Invalid stream length")
        cache = self.decoder.make_cache()
        keys = mx.zeros((0, self.decoder.args.hidden_size), dtype=mx.float32)
        values = mx.array([], dtype=mx.int32)
        last = None

        def advance(block):
            nonlocal keys, values, last
            ordinary, states = self.representations(mx.array([block]), cache)
            if last is not None:
                keys = mx.concatenate([keys, last], axis=0)
                values = mx.concatenate([values, mx.array(block[:1])])
            keys = mx.concatenate([keys, states[:-1]], axis=0)[-memory_length:]
            values = mx.concatenate([values, mx.array(block[1:], dtype=mx.int32)])[
                -memory_length:
            ]
            last = states[-1:]
            mx.eval(ordinary, keys, values, last)
            return ordinary[-1:]

        for i in range(0, len(prompt_ids), 128):
            h = advance(prompt_ids[i : i + 128])
        output = []
        for i in range(max_tokens):
            native = mx.softmax(self.decoder.project(h).astype(mx.float32), axis=-1)
            corrected = self.correct(
                native, last, keys, values, mx.ones((1, len(values)), dtype=mx.bool_)
            )
            token = mx.argmax(corrected, -1).item()
            output.append(token)
            if token in stop_ids or i + 1 == max_tokens:
                break
            h = advance([token])
        return output
