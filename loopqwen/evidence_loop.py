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
                (queries @ keys.swapaxes(-1, -2)) * (4 / keys.shape[-1]), axis=-1
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
    a, b = sentence_states(model, t, [*facts, question], depth)
    memory, first, _, last = net.read(a[None, :-1], b[None, :-1], a[None, -1], depth)
    mx.eval(memory, first, last)
    return EvidenceView(model, net, memory), first[0].tolist(), last[0].tolist()
