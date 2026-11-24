"""Independent requested-depth latency and late-pass precision/context interventions."""

from pathlib import Path
from statistics import median
from time import perf_counter
import mlx.core as mx
from ..data import pointer5, pointer6
from ..training.mechanism import requests
from ..training import pointer5 as kernel
from ..io import digest, save, read
from .mechanism import load, heads, packed, unpack


def latency(arm, step):
    m, t, st, w = load(arm, step)
    rows = [
        pointer6.ordered(r, "random0", 61091) for r in pointer5.rows("development")[:32]
    ]
    out = arm.output / f"latency-{step}.json"
    ident = [st["identity"], w, digest(Path(__file__))]
    if out.exists():
        if read(out)["identity"] != ident:
            raise ValueError("Latency identity mismatch")
        return
    results = []
    for n in (1, 2, 3):
        req = requests(rows, n, arm, t, {})
        for size in (1, 8):
            batches = [
                kernel.inputs(req[i : i + size]) for i in range(0, len(rows), size)
            ]
            mx.eval(batches)

            def predict(batch):
                x, last = batch
                if arm.kind == "heads":
                    h = next(m.block_states(x, 1))
                    layer = ([m.state_head, *m.extra_heads])[n - 1]
                    z = layer(kernel.normalize(h[mx.arange(len(last)), last]))
                else:
                    z = kernel.outputs(m, x, last, n, arm.kind)[-1]
                mx.eval(mx.argmax(z, axis=-1))

            predict(batches[0])
            times = []
            for batch in batches:
                start = perf_counter()
                predict(batch)
                times.append(perf_counter() - start)
            results.append(
                dict(
                    depth=n,
                    batch=size,
                    seconds=times,
                    median_ms=median(times) * 1000,
                    maps_s=len(rows) / sum(times),
                )
            )
    save(out, dict(identity=ident, results=results))


def late(arm, step):
    m, t, st, w = load(arm, step)
    out = arm.output / f"late-{step}.json.gz"
    ident = [st["identity"], w, digest(Path(__file__))]
    if out.exists():
        if unpack(out)["identity"] != ident:
            raise ValueError("Precision diagnostic identity mismatch")
        return
    rs = [
        (s, pointer6.ordered(r, s, 61091))
        for s in ("canonical", "random0")
        for r in pointer5.rows("development")[:96]
    ]
    preds = [
        dict(
            condition=s,
            id=r["example_id"],
            gold=r["states"],
            map=r["map"],
            start=r["start"],
        )
        for s, r in rs
    ]
    for precision in ("bf16", "fp32_block"):
        if precision == "fp32_block":
            for layer in m.text.layers[10:16]:
                layer.set_dtype(mx.float32)
        for i in range(0, len(rs), 8):
            chunk = [r for s, r in rs[i : i + 8]]
            x, last = kernel.inputs(requests(chunk, 1, arm, t, {}))
            h = m.prefix_hidden(x)
            if precision == "fp32_block":
                h = h.astype(mx.float32)
            scores = []
            for j in range(6):
                if j == 5 and precision == "bf16":
                    mask = (mx.arange(h.shape[1])[None, :] == last[:, None])[:, :, None]
                    contexts = [mx.where(mask, h, first), mx.where(mask, first, h)]
                    for name, v in zip(("fresh_context", "early_readout"), contexts):
                        patched = mx.argmax(
                            kernel.readout(m, m.shared_block(v), last), axis=-1
                        ).tolist()
                        for idx, val in enumerate(patched):
                            preds[i + idx][name] = val
                h = m.shared_block(h)
                if j == 0:
                    first = h
                scores.append(kernel.readout(m, h, last))
            p = mx.softmax(mx.stack(scores, axis=1), axis=-1).tolist()
            for idx, v in enumerate(p):
                preds[i + idx][precision] = v
    packed(out, dict(identity=ident, records=preds))
