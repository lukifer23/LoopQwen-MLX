"""A shared Qwen layer follows learned relations between caller-supplied passages."""

from collections import Counter
from math import log
import re

import mlx.core as mx
import mlx.nn as nn

from .evidence_loop import unit
from .operand_planner import adapted_attention_layer


class Terms:
    def __init__(self, text):
        words = re.findall(r"[^\W_]+", text.casefold())
        self.terms = set(words)
        self.text = " " + " ".join(words) + " "
        self.pairs = set(zip(words, words[1:]))


def links(question, titles, articles):
    """Measure directed overlap in the supplied passages, without corpus labels."""
    documents = [Terms(x) for x in articles]
    names = [Terms(re.sub(r"\s*\([^()]*\)\s*$", "", t)) for t in titles]
    q = Terms(question)
    counts = Counter(w for d in documents for w in d.terms)
    n = len(documents)
    weights = {w: 0.05 + log((n + 1) / (v + 1)) for w, v in counts.items()}
    absent = 0.05 + log(n + 1)
    for src in [q, *names, *documents]:
        src.total = sum(weights.get(w, absent) for w in sorted(src.terms))

    def affinity(a, b):
        common = sum(weights.get(w, absent) for w in sorted(a.terms & b.terms))
        return [
            common / max(0.05, a.total),
            common / max(0.05, b.total),
            len(a.pairs & b.pairs) / max(1, min(len(a.pairs), len(b.pairs))),
            float(bool(b.terms) and b.text in a.text),
        ]

    edges = []
    for title, body in [(q, q), *zip(names, documents, strict=True)]:
        edges.append(
            [
                affinity(body, name) + affinity(body, target) + affinity(title, target)
                for name, target in zip(names, documents, strict=True)
            ]
        )
    return mx.array(edges, mx.float32)


def features(model, tok, question, titles, articles):
    if len(titles) != len(articles) or len(titles) < 2:
        raise ValueError("Supply at least two matching titles and passages")
    encoded = [
        tok.encode(x, add_special_tokens=False) for x in [question, *titles, *articles]
    ]
    if not all(encoded):
        raise ValueError("Empty question, title or document")
    longest = max(map(len, encoded))
    ids = mx.array([r + [0] * (longest - len(r)) for r in encoded])
    length = mx.array([len(r) for r in encoded])
    embeddings = model.text.model.embed_tokens(ids).astype(mx.float32)
    mask = mx.arange(longest)[None, :] < length[:, None]
    means = unit((embeddings * mask[..., None]).sum(1) / length[:, None])
    n = len(titles)
    passages = mx.concatenate((means[1 : n + 1], means[n + 1 :]), axis=-1)
    query = mx.concatenate((means[:1], means[:1]), axis=-1)
    values = mx.concatenate((passages, query)).astype(mx.bfloat16)
    edge = links(question, titles, articles)
    mx.eval(values, edge)
    return values, edge


class LinkedReader(nn.Module):
    def __init__(self, model, rank=64, passes=2):
        super().__init__()
        self.passes = passes
        self.layer = adapted_attention_layer(model)
        width = model.args.hidden_size
        self.project = nn.Linear(2 * width, rank)
        self.relation = nn.Linear(12, rank)
        self.expand = nn.Linear(rank, width)
        self.key = nn.Linear(width, rank)
        self.query = nn.Linear(width, rank)
        self.link_score = nn.Linear(rank, 1)

    def __call__(self, values, edges):
        embedding = self.project(values.astype(mx.float32))
        prior = embedding[:, :-1]
        query = (0.2 * self.expand(mx.tanh(embedding[:, -1:]))).astype(mx.bfloat16)
        edge = mx.tanh(self.relation(edges))
        relation = edge[:, 0]
        scores = []
        # Both depths see the same edge tensors and supervise the same two labels.
        # At K=1 the second decision reuses the one-pass states.
        for hop in range(2):
            if hop < self.passes:
                context = (0.2 * self.expand(mx.tanh(prior + relation))).astype(
                    mx.bfloat16
                )
                hidden = mx.concatenate([context, query], axis=1)
                # This layer reads only the supplied passages and question. Pad one
                # inert position beyond the native short-query differentiation cutoff.
                n = hidden.shape[1]
                pad = max(0, 16 - n)
                mask = mx.where(mx.arange(n + pad) < n, 0.0, -1e9).astype(hidden.dtype)
                hidden = self.layer(
                    mx.pad(hidden, ((0, 0), (0, pad), (0, 0))),
                    mask=mask[None, None, None, :],
                )
                query = hidden[:, n - 1 : n]
                keys = unit(self.key(unit(hidden[:, : n - 1])))
            query_state = unit(self.query(unit(query)))
            score = (keys * query_state).mean(-1) * 8 + self.link_score(relation)[
                ..., 0
            ]
            if hop:
                score = score + mx.log(mx.maximum(1.0 - attention, 1e-5))
            scores.append(score)
            if not hop:
                attention = mx.softmax(score, axis=-1)
                relation = (attention[:, :, None, None] * edge[:, 1:]).sum(1) + edge[
                    :, 0
                ]
        return tuple(scores)
