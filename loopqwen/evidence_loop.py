"""A tied evidence read over sentence states from the pretrained Qwen span."""

import mlx.core as mx
import mlx.nn as nn


def unit(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-8)


def sentence_states(model, t, texts, depth):
    """One ordinary pass per sentence; the next hop re-enters the same six layers."""
    ids = [t.encode(text, add_special_tokens=False) for text in texts]
    length = max(map(len, ids))
    x = mx.array([v + [0] * (length - len(v)) for v in ids])
    h = model.shared_block(model.prefix_hidden(x))
    last = mx.array([len(v) - 1 for v in ids])
    idx = mx.arange(len(ids))
    first = h[idx, last]
    second = model.shared_block(h)[idx, last] if depth == 2 else first
    return unit(first), unit(second)


class EvidenceLoop(nn.Module):
    def __init__(self, width, vocab, rank):
        super().__init__()
        self.query = nn.Linear(width, rank)
        self.key = nn.Linear(width, rank)
        self.value = nn.Linear(width, rank)
        self.language = nn.Linear(width, rank)
        self.output = nn.Linear(rank, len(vocab))
        self.vocab = tuple(vocab)

    def read(self, facts, deeper, question, depth):
        keys = unit(self.key(facts))

        def retrieve(q):
            queries = unit(self.query(q))
            return mx.softmax(
                (queries @ keys.swapaxes(-1, -2))
                * (getattr(self, "temperature", 4) / keys.shape[-1]),
                axis=-1,
            )

        first = retrieve(question[:, None, :])
        # Marginalize the identity of the first supporting sentence. Each possible
        # next query comes solely from the intervening Qwen shared-block state.
        transition = retrieve(deeper)
        last = first @ transition if depth == 2 else first
        memory = (last @ self.value(facts))[:, 0]
        return memory, first[:, 0], transition, last[:, 0]

    def bias(self, hidden, memory):
        return self.output(mx.tanh(self.language(unit(hidden)) + memory[:, None, :]))


class EvidenceView:
    """Request-local logit residual, shared full-vocabulary decoder and caches."""

    def __init__(self, model, net, memory):
        self.model, self.net, self.memory = model, net, memory
        self.args = model.args

    def eval(self):
        self.model.eval()
        self.net.eval()

    def make_cache(self, *args, **kwargs):
        return self.model.make_cache(*args, **kwargs)

    def hidden(self, *args, **kwargs):
        return self.model.hidden(*args, **kwargs)

    def project(self, h):
        tiled = h[:, None, :] if h.ndim == 2 else h
        z = self.model.project(tiled).astype(mx.float32)
        z = z.at[..., mx.array(self.net.vocab)].add(self.net.bias(tiled, self.memory))
        return z[:, 0] if h.ndim == 2 else z


def prepare(model, t, net, facts, question, depth):
    encoder = pooled_states if isinstance(net, PooledEvidence) else sentence_states
    a, b = encoder(model, t, [*facts, question], depth)
    memory, first, _, last = net.read(a[None, :-1], b[None, :-1], a[None, -1], depth)
    mx.eval(memory, first, last)
    return EvidenceView(model, net, memory), first[0].tolist(), last[0].tolist()


def pooled_states(model, t, texts, depth):
    """Retain four ordered regions, so a final punctuation state cannot hide an argument."""
    ids = [t.encode(text, add_special_tokens=False) for text in texts]
    length = max(map(len, ids))
    x = mx.array([v + [0] * (length - len(v)) for v in ids])
    h = model.shared_block(model.prefix_hidden(x))
    second = model.shared_block(h) if depth == 2 else h
    # Fractional overlap with equal-width token regions supports any sentence length.
    lengths = mx.array([len(v) for v in ids])[:, None, None]
    bounds = lengths * (mx.arange(5) / 4)[None, :, None]
    pos = mx.arange(length)[None, None, :]
    weights = mx.maximum(
        0, mx.minimum(bounds[:, 1:], pos + 1) - mx.maximum(bounds[:, :-1], pos)
    )
    weights = weights / (lengths / 4)

    def pool(v):
        return (weights @ unit(v)).reshape(len(ids), -1)

    return pool(h), pool(second)


class PooledEvidence(EvidenceLoop):
    def __init__(self, width, vocab, rank):
        super().__init__(width, vocab, rank)
        self.key = nn.Linear(4 * width, rank)
        self.query = nn.Linear(4 * width, rank)
        self.value = nn.Linear(4 * width, rank)
        self.content = nn.Linear(rank, len(vocab))
        self.gate = nn.Linear(rank, len(vocab))
        self.temperature = 10
        for label in ("first", "deep"):
            setattr(self, label + "_mean", mx.zeros(4 * width))
            setattr(self, label + "_scale", mx.ones(4 * width))

    def calibrate(self, first, deeper):
        # Statistics describe training sentences only, with a variance floor.
        for label, x in (("first", first), ("deep", deeper)):
            setattr(self, label + "_mean", mx.mean(x, axis=0))
            setattr(self, label + "_scale", mx.rsqrt(mx.var(x, axis=0) + 0.0001))
        self.freeze(keys=["first_mean", "first_scale", "deep_mean", "deep_scale"])

    def read(self, facts, deeper, question, depth):
        return super().read(
            (facts - self.first_mean) * self.first_scale,
            (deeper - self.deep_mean) * self.deep_scale,
            (question - self.first_mean) * self.first_scale,
            depth,
        )

    def bias(self, hidden, memory):
        language = mx.tanh(self.language(unit(hidden)))
        return (
            self.output(language)
            + mx.sigmoid(self.gate(language)) * self.content(memory)[:, None, :]
        )


def passage(t, ids):
    """Read the last user turn as newline-separated statements followed by a question."""
    if 248045 not in ids or 248046 not in ids:
        return None
    # Only the user message preceding the currently open assistant turn qualifies.
    text = t.decode(ids)
    prefix, sep, tail = text.rpartition("<|im_start|>user\n")
    if not sep:
        return None
    body, end, assistant = tail.partition("<|im_end|>")
    if not end or assistant.strip() != "<|im_start|>assistant\n<think>":
        return None
    lines = [v.strip() for v in body.splitlines() if v.strip()]
    if len(lines) < 3 or not lines[-1].endswith("?"):
        return None
    return lines[:-1], lines[-1]
