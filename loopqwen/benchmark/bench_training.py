import mlx.core as mx
import mlx.nn as nn
from .bench_gdn import timed, relative
from ..data.tokenize import batch
from ..training.losses import loss_sum


def bench_training(model, rows, c, n):
    from ..kernels.gdn_backend import select

    model.train()
    x, pos, y = batch(rows, c["response_only"])
    h = model.hidden(x)
    mx.eval(h)

    def standard(h):
        logits = model.project(h).reshape(-1, model.args.vocab_size)[pos]
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="sum")

    def selected(h):
        return mx.fast.cross_entropy(
            model.project(h.reshape(-1, h.shape[-1])[pos]), y
        ).sum()

    loss_bench = {}
    for fn in (standard, selected):
        vg = mx.value_and_grad(fn)
        mx.clear_cache()
        mx.reset_peak_memory()
        loss_bench[fn.__name__] = dict(
            forward_ms=timed(lambda: fn(h), n),
            combined_ms=timed(lambda: vg(h), n),
            peak_bytes=mx.get_peak_memory(),
        )
    a, ga = mx.value_and_grad(standard)(h)
    b, gb = mx.value_and_grad(selected)(h)
    err = relative(gb, ga)
    assert abs((a - b).item()) / y.size < 0.01 and err < 0.02
    result = []
    for backend, ckpt in (("reference", True), ("native", True), ("native", False)):
        select(model, backend)
        model.checkpointing = ckpt
        mx.clear_cache()
        mx.reset_peak_memory()
        vg = nn.value_and_grad(model, loss_sum)
        forward = timed(lambda: loss_sum(model, x, pos, y), n)
        combined = timed(lambda: vg(model, x, pos, y), n)
        result.append(
            dict(
                backend=backend,
                checkpointing=ckpt,
                forward_ms=forward,
                combined_ms=combined,
                backward_estimate_ms=combined - forward,
                peak_bytes=mx.get_peak_memory(),
            )
        )
    return dict(
        rows=result,
        loss=loss_bench,
        loss_gradient_error=err,
        input_shape=x.shape,
        labels=y.size,
    )
