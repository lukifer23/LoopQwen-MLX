"""Compare two BF16 fusion rules on identical held-out development examples."""

import gc
from pathlib import Path

import mlx.core as mx

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..export import export
from ..export_runtime import load_export, verify
from ..io import save
from .phase2 import nll, predictions


def diagnostics(model, tok, docs, language):
    top = []
    for r in docs:
        x = mx.array([r["ids"][:-1]])
        pred = mx.argmax(model(x)[:, r["prefix"] - 1 :], -1)
        top.extend(pred.flatten().tolist())
    return dict(
        answer_nll=nll(model, docs),
        language=nll(model, language),
        greedy=predictions(model, tok, docs, 12),
        scored_top1=top,
    )


def audit(name):
    checkpoint = Path("checkpoints") / name
    docs = rows("validation", "answer")[:30]
    lang = rows("language_validation", "language")[:24]
    model, tok, _ = load_trained(checkpoint)
    base = diagnostics(model, tok, docs, lang)
    del model
    result = {"adapter": base}
    for method in ("stock", "fp32"):
        destination = Path("exports") / (name + "_" + method)
        if not destination.exists():
            export(checkpoint, destination, method)
        verify(destination)
        model, tok = load_export(destination)
        r = diagnostics(model, tok, docs, lang)
        r["top1_disagreement"] = sum(
            a != b for a, b in zip(base["scored_top1"], r["scored_top1"])
        ) / len(base["scored_top1"])
        r["generation_disagreement"] = sum(
            a["text"] != b["text"]
            for a, b in zip(base["greedy"]["records"], r["greedy"]["records"])
        ) / len(docs)
        result[method] = r
        del model
        gc.collect()
        mx.clear_cache()
    save("outputs/phase2/merge_audit.json", result)
    return result


if __name__ == "__main__":
    import sys

    audit(sys.argv[1])
