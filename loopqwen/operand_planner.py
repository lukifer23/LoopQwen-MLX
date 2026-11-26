"""A shared native Qwen layer chooses arithmetic operations and their operand references."""

from copy import deepcopy
from fractions import Fraction
from math import copysign, log1p

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from .arithmetic_actions import CONSTANTS, OPS, as_steps, mentions
from .evidence_loop import unit

LAYER = 15
STEPS = 6


def numeric_features(value):
    x = float(Fraction(value))
    return [
        copysign(log1p(abs(x)), x) / 6,
        1 / (1 + abs(x)),
        float(x.is_integer()),
        float(x < 0),
    ]


def encode_document(model, tok, text, question):
    """The decoder sees the common full prompt. Gather only number spans inside the user question."""
    encoded = tok.backend_tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids
    if ids != tok.encode(text, add_special_tokens=False):
        raise ValueError("Token alignment differs")
    start = text.rindex(question)
    spans = mentions(question)
    offsets = encoded.offsets
    content = [
        i for i, (a, b) in enumerate(offsets) if b > start and a < start + len(question)
    ]
    h = model._execute(
        model.text.model.embed_tokens(mx.array([ids])), [(i, 0) for i in range(LAYER)]
    )[0]
    h = unit(h)
    last = content[-1]
    query = mx.concatenate([h[last], h[max(content[0], last - 15) : last + 1].mean(0)])
    rows = []
    for a, b, _ in spans:
        indices = [
            i
            for i in content
            if offsets[i][1] > a + start and offsets[i][0] < b + start
        ]
        if not indices:
            raise ValueError("Missing number token span")
        first, end = indices[0], indices[-1]
        rows.append(
            mx.concatenate(
                [h[end], h[max(content[0], first - 5) : min(last + 1, end + 7)].mean(0)]
            )
        )
    features = mx.stack([query, *rows]).astype(mx.bfloat16)
    mx.eval(features)
    return features, [s[2] for s in spans]


def slot(ref, n):
    prefix, i = ref[0], int(ref[1:])
    if prefix == "c":
        return i
    if prefix == "n":
        return len(CONSTANTS) + i
    if prefix == "r":
        return len(CONSTANTS) + n + i
    raise ValueError("Unknown operand reference")


def reference(index, n):
    if index < len(CONSTANTS):
        return "c" + str(index)
    index -= len(CONSTANTS)
    return "n" + str(index) if index < n else "r" + str(index - n)


def adapted_attention_layer(model, layer_index=LAYER, rank=8):
    """A native layer with the frozen donor arrays and four trainable projection adapters."""
    original = model.text.layers[layer_index]
    layer = deepcopy(
        original, {id(v): v for _, v in tree_flatten(original.parameters())}
    )
    layer.freeze()
    for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(
            layer.self_attn,
            k,
            LoRALinear.from_base(
                getattr(layer.self_attn, k), r=rank, scale=2, dropout=0
            ),
        )
    adapted = {k for k, v in tree_flatten(layer.trainable_parameters())}
    expected = {
        f"self_attn.{name}.lora_{part}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        for part in ("a", "b")
    }
    if adapted != expected:
        raise ValueError("Unexpected trainable native projections")
    return layer


class OperandPlanner(nn.Module):
    def __init__(self, model, rank, depth):
        super().__init__()
        w = model.args.hidden_size
        self.layer = adapted_attention_layer(model)
        self.depth = depth
        self.rank = rank
        self.compress = nn.Linear(2 * w, rank)
        self.numeric = nn.Linear(4, rank)
        self.constants = mx.random.normal((len(CONSTANTS), rank)) * 0.1
        self.operation = mx.random.normal((4, rank)) * 0.1
        self.combine = nn.Linear(3 * rank, rank)
        self.expand = nn.Linear(rank, w)
        self.key = nn.Linear(w, rank)
        self.left = nn.Linear(w, 4 * rank)
        self.right = nn.Linear(w, 4 * rank)
        self.op = nn.Linear(w, 4)
        self.halt = nn.Linear(w, 1)

    def __call__(self, features, values, lengths, history, used):
        b, n = features.shape[0], features.shape[1] - 1
        seq = mx.tanh(self.compress(features.astype(mx.float32)))
        constants = mx.broadcast_to(self.constants, (b, *self.constants.shape))
        items = mx.concatenate([constants, seq[:, 1:]], axis=1) + self.numeric(values)
        base = items.shape[1]
        items = mx.pad(items, ((0, 0), (0, STEPS), (0, 0)))
        batch = mx.arange(b)
        for i in range(history.shape[1]):
            op, l, r = history[:, i, :].T
            x = mx.concatenate(
                [items[batch, l], items[batch, r], self.operation[op]], axis=-1
            )
            items = items.at[:, base + i, :].add(mx.tanh(self.combine(x)))
        valid = mx.concatenate(
            [
                mx.ones((b, len(CONSTANTS)), dtype=mx.bool_),
                mx.arange(n)[None, :] < lengths[:, None],
                mx.arange(STEPS)[None, :] < used[:, None],
            ],
            axis=-1,
        )
        active = mx.concatenate([valid, mx.ones((b, 1), dtype=mx.bool_)], axis=1)
        h = (0.2 * self.expand(mx.concatenate([items, seq[:, :1]], axis=1))).astype(
            mx.bfloat16
        )
        s = h.shape[1]
        mask = mx.where(
            (mx.arange(s)[:, None] >= mx.arange(s)[None, :])[None, None, :, :]
            & active[:, None, None, :],
            0.0,
            -1e9,
        ).astype(h.dtype)
        for _ in range(self.depth):
            h = self.layer(h, mask=mask)
        q = unit(h[:, -1])
        k = unit(self.key(unit(h[:, :-1])))
        l = unit(self.left(q).reshape(b, 4, self.rank)) @ k.transpose(0, 2, 1)
        r = unit(self.right(q).reshape(b, 4, self.rank)) @ k.transpose(0, 2, 1)
        scores = self.op(q)[:, :, None, None] + (
            l[:, :, :, None] + r[:, :, None, :]
        ) / (self.rank**0.5)
        good = valid[:, None, :, None] & valid[:, None, None, :]
        return mx.where(good, scores, -1e8), self.halt(q)[:, 0]


def padded(features, values, n):
    h = mx.pad(features, ((0, n + 1 - features.shape[0]), (0, 0)))
    v = mx.array(
        [numeric_features(x) for x in (*CONSTANTS, *values)]
        + [[0.0] * 4] * (n - len(values))
    )
    return h, v


def rollout(planner, features, values):
    n = max(16, len(values))
    h, v = padded(features, values, n)
    actions = []
    indices = []
    for step in range(STEPS):
        history = mx.array(indices, dtype=mx.int32).reshape(1, step, 3)
        logits, halt = planner(
            h[None], v[None], mx.array([len(values)]), history, mx.array([step])
        )
        act = mx.argmax(logits.reshape(-1))
        mx.eval(act, halt)
        width = logits.shape[-1]
        op, rem = divmod(act.item(), width * width)
        l, r = divmod(rem, width)
        indices.append([op, l, r])
        actions.append([OPS[op], reference(l, n), reference(r, n)])
        if halt.item() >= 0:
            break
    return as_steps(values, actions), actions
