"""Measure the second visit on preselected original training responses after static merging."""

from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from ..cross_view import CrossViewAssistant
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import read
from .cross_view_study import tokenizer
from .native_function_data import token_pair


def sample(views):
    groups = defaultdict(list)
    for w in views:
        key = w["kind"] if w["group"] == "database" else w["group"]
        groups[key].append(w)
    assert len(groups) == 7
    return [
        (kind, w)
        for kind, items in sorted(groups.items())
        for w in sorted(items, key=lambda r: fingerprint([518391, r["id"]]))[:4]
    ]


def measure(directory, root):
    if (root / "merged-training-visits.json.gz").is_file():
        return
    s = read(root / "registration.json")
    net = CrossViewAssistant(directory, root)._tool_model
    tok = tokenizer(s)
    rows = []
    close = tok.convert_tokens_to_ids("</think>")
    for kind, w in sample(read(s["training"])["examples"]):
        prefix, labels = token_pair(tok, w)
        j = labels.index(close) + 1
        while j < len(labels) - 1 and not tok.decode([labels[j]]).strip():
            j += 1
        pos = sorted({0, j, len(labels) - 1})
        seq = mx.array([prefix + labels[:-1]])
        indices = mx.array([len(prefix) - 1 + k for k in pos])
        first = net._execute(net.text.model.embed_tokens(seq), net.sites[:-1])
        second = net._execute(first, net.sites[-1:])
        result = {
            "group": kind,
            "view": w["id"],
            "positions": pos,
            "target": [labels[k] for k in pos],
        }
        for key, state in (("first", first), ("second", second)):
            logits = net.project(net.text.model.norm(state[0, indices])).astype(
                mx.float32
            )
            nll = (
                mx.logsumexp(logits, axis=-1)
                - logits[mx.arange(len(pos)), mx.array(result["target"])]
            )
            result[key] = {
                "nll": nll.tolist(),
                "top": mx.argmax(logits, axis=-1).tolist(),
            }
        rows.append(result)
    packed(Path(root) / "merged-training-visits.json.gz", rows)
