from dataclasses import replace

import mlx.core as mx

from .evaluation.generation_eval import generate
from .io import save
from .looped_model import LoopedModel
from .model_loader import load_base
from .recurrence import Span


def cache_check():
    base, tok, _ = load_base()
    model = LoopedModel(base)
    model.eval()
    rows = []
    for k in (1, 2, 3):
        model.span = replace(Span(), passes=k)
        for prompt in (
            "Birds migrate south in winter.",
            " ".join(str(i) for i in range(20)),
        ):
            ids = mx.array([tok.encode(prompt)])
            full = model(ids).astype(mx.float32)
            mx.eval(full)
            for chunk in (1, 7):
                cache = model.make_cache()
                assert len({id(s) for s in cache.slots.values()}) == len(model.sites)
                parts = []
                for i in range(0, ids.size, chunk):
                    part = model(ids[:, i : i + chunk], cache=cache).astype(mx.float32)
                    mx.eval(part)
                    parts.append(part)
                cached = mx.concatenate(parts, axis=1)
                error = cached - full
                relative = (
                    mx.sqrt(mx.mean(error**2, -1) / mx.mean(full**2, -1)).max().item()
                )
                maximum = mx.abs(error).max().item()
                assert all(
                    s.offset == ids.size
                    for s in cache.slots.values()
                    if hasattr(s, "offset")
                )
                row = dict(
                    k=k,
                    length=ids.size,
                    chunk=chunk,
                    max_abs=maximum,
                    worst_position_relative_rms=relative,
                    top1_agreement=mx.mean(
                        mx.argmax(full, -1) == mx.argmax(cached, -1)
                    ).item(),
                )
                rows.append(row)
                if relative > (0.04 if k == 3 else 0.03) or maximum > 1.5:
                    raise AssertionError(row)
    model.span = Span()
    demo = generate(model, tok, "The sum of two and three is", 12)
    result = dict(rows=rows, generation=demo)
    save("outputs/cache.json", result)
    return result


def fp32_diagnostic():
    base, tok, _ = load_base()
    base.set_dtype(mx.float32)
    mx.eval(base.parameters())
    mx.clear_cache()
    model = LoopedModel(base, Span(passes=3))
    model.eval()
    ids = mx.array([tok.encode(" ".join(str(i) for i in range(20)))])
    full = model(ids)
    mx.eval(full)
    cache = model.make_cache()
    pieces = []
    for i in range(ids.size):
        part = model(ids[:, i : i + 1], cache)
        mx.eval(part)
        pieces.append(part)
    delta = mx.concatenate(pieces, 1) - full
    result = dict(
        max_abs=mx.max(mx.abs(delta)).item(),
        relative_rms=mx.sqrt(mx.mean(delta**2, -1) / mx.mean(full**2, -1)).max().item(),
    )
    if result["relative_rms"] > 0.0001:
        raise AssertionError(result)
    save("outputs/cache_fp32.json", result)
    return result
