"""A learned sentence representation refined solely by re-running one native Qwen layer."""

from copy import deepcopy

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRALinear

from .evidence_loop import unit

LAYER = 15


def prefix_regions(model, t, texts, _depth):
    ids = [t.encode(x, add_special_tokens=False) for x in texts]
    n = max(map(len, ids))
    inputs = mx.array([v + [0] * (n - len(v)) for v in ids])
    h = model._execute(
        model.text.model.embed_tokens(inputs), [(i, 0) for i in range(LAYER)]
    )
    lengths = mx.array([len(v) for v in ids])[:, None, None]
    bounds = lengths * (mx.arange(5) / 4)[None, :, None]
    pos = mx.arange(n)[None, None, :]
    w = mx.maximum(
        0, mx.minimum(bounds[:, 1:], pos + 1) - mx.maximum(bounds[:, :-1], pos)
    ) / (lengths / 4)
    pooled = (w @ unit(h)).reshape(len(ids), -1)
    return pooled, pooled


class NativeEvidence(nn.Module):
    def __init__(self, model, rank, vocab):
        super().__init__()
        width = model.args.hidden_size
        self.vocab = tuple(vocab)
        source = model.text.layers[LAYER]
        if source.is_linear:
            raise ValueError("Required pretrained full attention layer")
        # Share the pinned immutable arrays with the ordinary full-prompt decoder.
        self.layer = deepcopy(
            source, {id(v): v for _, v in tree_flatten(source.parameters())}
        )
        self.layer.freeze()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            original = getattr(self.layer.self_attn, name)
            setattr(
                self.layer.self_attn,
                name,
                LoRALinear.from_base(original, r=8, scale=2, dropout=0),
            )
        self.compress = nn.Linear(4 * width, rank)
        self.expand = nn.Linear(rank, width)
        self.readout = nn.Linear(width, rank)
        self.language = nn.Linear(width, rank)
        self.format = nn.Linear(rank, len(vocab))
        self.content = nn.Linear(rank, len(vocab))
        self.gate = nn.Linear(rank, len(vocab))
        self.support_keys = nn.Linear(4 * width, rank)
        self.support_queries = [nn.Linear(width, rank) for _ in range(3)]
        self.center = mx.zeros(4 * width)
        self.scale = mx.ones(4 * width)
        self.freeze(keys=["center", "scale"], recurse=False)
        trainable = dict(tree_flatten(self.trainable_parameters()))
        expected = {
            f"layer.self_attn.{proj}.lora_{part}"
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
            for part in ("a", "b")
        }
        if {name for name in trainable if name.startswith("layer.")} != expected:
            raise ValueError(
                "Native Qwen trainability must match the four low-rank projections"
            )

    def states(self, facts, q, depth):
        x = (mx.concatenate([facts, q[:, None]], axis=1) - self.center) * self.scale
        h = (0.2 * self.expand(mx.tanh(self.compress(x)))).astype(mx.bfloat16)
        mask = create_attention_mask(h)
        queries = []
        for _ in range(depth):
            h = self.layer(h, mask=mask)
            queries.append(unit(h[:, -1]))
        return queries, x[:, :-1]

    def bias(self, h, memory):
        lang = mx.tanh(self.language(unit(h)))
        return (
            self.format(lang)
            + mx.sigmoid(self.gate(lang)) * self.content(memory)[:, None, :]
        )

    def auxiliary(self, queries, facts, support):
        keys = unit(self.support_keys(facts))
        losses = []
        for i, proj in enumerate(self.support_queries):
            query = unit(proj(queries[min(i, len(queries) - 1)]))
            logits = 10 * mx.mean(keys * query[:, None], axis=-1)
            losses.append(
                nn.losses.cross_entropy(logits, support[:, i], reduction="mean")
            )
        return sum(losses) / len(losses)
