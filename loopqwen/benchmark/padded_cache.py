"""Validate batched left padding against independent, unpadded cached requests."""

from dataclasses import replace

import mlx.core as mx

from ..data.v2 import rows
from ..evaluation.phase2 import predictions
from ..io import save
from ..looped_model import LoopedModel
from ..model_loader import load_base


def run():
    base, tok, _ = load_base()
    model = LoopedModel(base)
    model.eval()
    samples = rows("validation", "trace")[:15]
    metrics = {}
    docs = [r["ids"][: r["prefix"]] for r in samples[:4]]
    width = max(map(len, docs))
    pads = [width - len(d) for d in docs]
    for dtype in ("bf16", "fp32"):
        if dtype == "fp32":
            base.set_dtype(mx.float32)
            mx.eval(base.parameters())
        for k in (1, 2, 3):
            model.span = replace(model.span, passes=k)
            cache = model.make_cache(left_padding=pads)
            single = [model.make_cache() for _ in docs]
            x = mx.array([[0] * p + v for p, v in zip(pads, docs)])
            seq = [mx.array([v]) for v in docs]
            errors = []
            agree = []
            for t in range(5):
                a = model.project(model.hidden(x, cache)[:, -1]).astype(mx.float32)
                b = mx.concatenate(
                    [
                        model.project(model.hidden(z, c)[:, -1])
                        for z, c in zip(seq, single)
                    ]
                ).astype(mx.float32)
                errors.append(mx.sqrt(mx.mean((a - b) ** 2) / mx.mean(b**2)).item())
                agree.append(mx.mean(mx.argmax(a, -1) == mx.argmax(b, -1)).item())
                if t == 0:
                    cache.finish_left_prefill()
                x = mx.argmax(b, -1)[:, None]
                seq = [token[None] for token in x]
            if max(errors) > (0.03 if dtype == "bf16" else 0.0001):
                raise AssertionError((dtype, k, errors))
            metrics[f"{dtype}/{k}"] = dict(
                relative_rms=errors, top1=agree, padding=pads
            )
        if dtype == "bf16":
            model.span = replace(model.span, passes=2)
            for p in (False, True):
                r = predictions(model, tok, samples, 32, padded=p)
                metrics[f"latency/{p}"] = dict(
                    seconds=r["seconds"], records=r["records"]
                )
    save("outputs/phase2/padded.json", metrics)


if __name__ == "__main__":
    run()
