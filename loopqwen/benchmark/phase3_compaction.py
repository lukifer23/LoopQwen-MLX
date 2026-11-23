"""Measure optional active-lane filtering against the registered fixed-batch path."""

import mlx.core as mx

from .. import checkpoint
from ..data.v2 import rows
from ..evaluation.phase2 import predictions
from ..io import read, save
from pathlib import Path


def run():
    arm = read("outputs/phase3/selection.json")["selected"]
    root = Path("checkpoints") / ("p3_" + arm)
    m, t, _ = checkpoint.load_trained(
        root / read(root / "targets-100000.json")["directory"]
    )
    docs = rows("validation", "trace")[:30]
    predictions(m, t, docs[:4], 12, padded=True)
    results = []
    for compact in (False, True, True, False):
        r = predictions(m, t, docs, 192, padded=True, compact=compact, detailed=True)
        results.append(
            dict(
                compact=compact,
                seconds=r["seconds"],
                positions=sum(b["executed_positions"] for b in r["batches"]),
                correct=r["correct"],
                records=r["records"],
            )
        )
    m.set_dtype(mx.float32)
    doc = docs[:4]
    width = max(r["prefix"] for r in doc)
    pads = [width - r["prefix"] for r in doc]
    ids = mx.array([[0] * p + r["ids"][: r["prefix"]] for p, r in zip(pads, doc)])
    caches = [m.make_cache(left_padding=pads) for _ in range(2)]
    for c in caches:
        mx.eval(m.hidden(ids, c))
        c.finish_left_prefill()
    keep = mx.array([1, 3])
    for slot in caches[1].slots.values():
        slot.filter(keep)
    token = mx.array([[3], [4], [5], [6]])
    errors = []
    for _ in range(4):
        full = m(token, caches[0])[:, 0].astype(mx.float32)
        small = m(token[keep], caches[1])[:, 0].astype(mx.float32)
        ref = full[keep]
        errors.append(mx.sqrt(mx.mean((small - ref) ** 2) / mx.mean(ref**2)).item())
        token = mx.argmax(full, -1)[:, None]
    if max(errors) > 1e-4:
        raise AssertionError(errors)
    save(
        "outputs/phase3/compaction.json",
        dict(fp32_filtered_errors=errors, timing=results),
    )


if __name__ == "__main__":
    run()
