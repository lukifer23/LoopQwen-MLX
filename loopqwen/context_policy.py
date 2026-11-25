"""Learn which recurrent attention reader to use from the actual prompt token stream."""

from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .attention_memory import AttentionReader, PassageScorer
from .io import digest, read


def features(embed, ids):
    n = max(1, len(ids) - 64)
    # Only the routing classifier samples. Both attention scorers and the decoder read all IDs.
    positions = [i * (n - 1) // 63 for i in range(64)]
    observed = embed(mx.array([ids[i] for i in positions] + ids[-64:])).astype(
        mx.float32
    )
    mean = mx.concatenate([observed[:64].mean(0), observed[64:].mean(0)])
    return mean * mx.rsqrt(mx.mean(mean * mean) + 1e-7)


class ContextPolicy:
    def __init__(self, ordinary, prose, classifier):
        self.ordinary = ordinary
        self.prose = prose
        self.classifier = classifier

    def choose(self, model, ids, cap):
        from .deliberation import turn_open

        if not turn_open(ids, 248068, 248069):
            return self.ordinary, "ordinary"
        if len(ids) < 1024 and cap > 256:
            return self.ordinary, "ordinary"
        evidence = features(model.text.model.embed_tokens, ids)
        kind = self.classifier(evidence).argmax().item()
        if len(ids) < 1024:
            return (
                (model, "short_recurrent") if kind == 0 else (self.ordinary, "ordinary")
            )
        if kind == 2:
            return self.prose, "prose_recurrent"
        if kind == 1:
            return model, "register_recurrent"
        return self.ordinary, "ordinary"


def attach(model, folder):
    from .deliberation import ordinary_view

    folder = Path(folder)
    spec = read(folder / "context-policy.json")
    for k, h in spec["files"].items():
        if digest(folder / k) != h:
            raise ValueError("Context policy checkpoint differs")
    original = ordinary_view(model)
    shared = {id(v): v for _, v in tree_flatten(model.parameters())}
    prose = deepcopy(model, shared)
    weights = mx.load(str(folder / "prose-adapter.safetensors"))
    if set(weights) != set(dict(tree_flatten(prose.trainable_parameters()))):
        raise ValueError("Prose adapter projection set changed")
    prose.load_weights(list(weights.items()), strict=False)
    prose.eval()
    scorer = PassageScorer(model.args.hidden_size, **spec["architecture"])
    scorer.load_weights(str(folder / "prose-scorer.safetensors"), strict=True)
    scorer.eval()
    object.__setattr__(
        prose, "attention_reader", AttentionReader(scorer, spec["gain"], 1024)
    )
    classifier = nn.Linear(2 * model.args.hidden_size, 3)
    classifier.load_weights(str(folder / "selector.safetensors"), strict=True)
    classifier.eval()
    object.__setattr__(
        model, "context_policy", ContextPolicy(original, prose, classifier)
    )
