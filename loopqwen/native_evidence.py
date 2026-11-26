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


class AnchoredNativeEvidence(NativeEvidence):
    """Refine a single question state by native attention to stable sentence tokens."""

    def __init__(self, model, rank, vocab):
        super().__init__(model, rank, vocab)
        del self.support_keys
        del self.support_queries

    def states(self, facts, q, depth):
        x = (mx.concatenate([facts, q[:, None]], axis=1) - self.center) * self.scale
        h = (0.2 * self.expand(mx.tanh(self.compress(x)))).astype(mx.bfloat16)
        source, query = h[:, :-1], h[:, -1:]
        mask = create_attention_mask(h)
        queries = []
        inputs = []
        # Deterministic step codes distinguish the iterations without giving the
        # recurrent arm an extra trainable vector per pass.
        freq = mx.exp(
            -mx.arange(query.shape[-1], dtype=mx.float32) * 6 / query.shape[-1]
        )
        for i in range(depth):
            state = (query + 0.04 * mx.sin(i * freq)).astype(query.dtype)
            h = mx.concatenate([source, state], axis=1)
            inputs.append(h)
            query = self.layer(h, mask=mask)[:, -1:]
            queries.append(unit(query[:, 0]))
        return queries, inputs

    def native_scores(self, h):
        # Reconstruct only the final query row from the very Q/K tensors, RoPE
        # positions and scaling that the native scaled-dot-product operator uses.
        attn = self.layer.self_attn
        x = self.layer.input_layernorm(h)
        b, n, _ = x.shape
        q, _ = mx.split(
            attn.q_proj(x).reshape(b, n, attn.num_attention_heads, -1),
            2,
            axis=-1,
        )
        q = attn.q_norm(q).transpose(0, 2, 1, 3)
        k = attn.k_norm(
            attn.k_proj(x).reshape(b, n, attn.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        q = attn.rope(q)[:, :, -1:]
        k = attn.rope(k)
        k = mx.repeat(k, attn.num_attention_heads // attn.num_key_value_heads, axis=1)
        return mx.sum(q.astype(mx.float32) * k.astype(mx.float32), axis=-1) * attn.scale

    def auxiliary(self, queries, inputs, support):
        del queries
        logits = [self.native_scores(h) for h in inputs]
        loss = []
        for i in range(3):
            # Separate supervised head pairs allow the one-pass twin to assign
            # different mass to each of the three annotated facts at the same exit.
            z = logits[min(i, len(logits) - 1)][:, 2 * i : 2 * i + 2, :]
            labels = mx.broadcast_to(support[:, i : i + 1], z.shape[:-1])
            loss.append(nn.losses.cross_entropy(z, labels, reduction="mean"))
        return sum(loss) / 3


def load_native_reader(model, folder, expected):
    from . import REVISION
    from .io import digest, read

    spec = read(folder / "reader.json")
    path = folder / "reader.safetensors"
    if (
        digest(path) != spec["weights"]
        or spec["weights"] != expected
        or spec["base"] != REVISION
    ):
        raise ValueError("Wrong native reader lineage")
    architectures = {
        "native": NativeEvidence,
        "anchored_native": AnchoredNativeEvidence,
    }
    architecture = spec.get("architecture", "native")
    if architecture not in architectures:
        raise ValueError("Unsupported native evidence architecture")
    net = architectures[architecture](model, spec["rank"], spec["vocab"])
    weights = mx.load(str(path))
    expected_arrays = dict(
        tree_flatten(net.trainable_parameters()), center=net.center, scale=net.scale
    )
    if weights.keys() != expected_arrays.keys() or any(
        v.shape != expected_arrays[key].shape for key, v in weights.items()
    ):
        raise ValueError("Incomplete or incompatible learned native tensors")
    net.load_weights(list(weights.items()), strict=False)
    net.eval()
    return net


def native_view(model, t, net, facts, question, depth):
    from .evidence_loop import EvidenceView

    features, _ = prefix_regions(model, t, [*facts, question], 1)
    states, inputs = net.states(features[None, :-1], features[None, -1], depth)
    return EvidenceView(model, net, net.readout(states[-1])), inputs
