from statistics import median
from time import perf_counter

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update

from ..io import save
from ..kernels.gdn_backend import native_update


def relative(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return mx.sqrt(mx.sum((a - b) ** 2) / (mx.sum(b**2) + 1e-30)).item()


def timed(fn, n=5):
    mx.eval(fn())
    times = []
    for _ in range(n):
        mx.synchronize()
        start = perf_counter()
        mx.eval(fn())
        times.append(perf_counter() - start)
    return median(times) * 1000


def bench_gdn():
    mx.random.seed(42)

    def ref(*args):
        return gated_delta_update(*args, use_kernel=False)

    rows = []
    for length, heads, dim, dtype in [
        (7, 2, 32, mx.float32),
        (32, 16, 128, mx.bfloat16),
        (64, 16, 128, mx.bfloat16),
    ]:

        def rand(shape):
            return (mx.random.normal(shape) * 0.12).astype(dtype)

        vector = (1, length, heads, dim)
        gate = vector[:-1]
        args = [rand(vector) for _ in range(3)] + [
            rand(gate),
            rand(gate),
            rand((heads,)),
            rand((heads,)),
            rand((1, heads, dim, dim)).astype(mx.float32),
        ]
        cot = [mx.random.normal(vector), mx.random.normal(args[-1].shape)]

        def objective(fn, *xs):
            y, s = fn(*xs)
            return (y * cot[0]).sum() + (s * cot[1]).sum()

        grads = []
        outputs = []
        perf = {}
        for fn in (ref, native_update):
            grad = mx.value_and_grad(
                lambda *xs: objective(fn, *xs), argnums=list(range(len(args)))
            )
            mx.clear_cache()
            mx.reset_peak_memory()
            forward = timed(lambda: fn(*args))
            combined = timed(lambda: grad(*args))
            outputs.append(fn(*args))
            grads.append(grad(*args)[1])
            perf[fn.__name__] = dict(
                forward_ms=forward,
                value_grad_ms=combined,
                backward_estimate_ms=combined - forward,
                peak_bytes=mx.get_peak_memory(),
            )
        errors = [relative(a, b) for a, b in zip(outputs[1], outputs[0])]
        grad_errors = [relative(a, b) for a, b in zip(grads[1], grads[0])]
        row = dict(
            length=length,
            heads=heads,
            dim=dim,
            dtype=str(dtype),
            output_state_error=errors,
            gradient_errors=grad_errors,
            timings=perf,
        )
        if max(errors) > 0.01 or max(grad_errors) > 0.025:
            raise AssertionError(row)
        if dtype == mx.float32:
            direction = mx.zeros_like(args[0])
            direction[0, 0, 0, 0] = 1
            eps = 0.002
            plus = objective(native_update, args[0] + direction * eps, *args[1:])
            minus = objective(native_update, args[0] - direction * eps, *args[1:])
            numerical = ((plus - minus) / (2 * eps)).item()
            analytic = grads[1][0][0, 0, 0, 0].item()
            assert abs(numerical - analytic) < 0.002, (numerical, analytic)
            row["finite_difference_q"] = [numerical, analytic]
        rows.append(row)
    save("outputs/gdn.json", rows)
    return rows


def check_projection_gradients():
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from ..kernels.gdn_backend import select
    from ..looped_model import LoopedModel
    from ..model_loader import load_base

    model = LoopedModel(load_base()[0])
    model.train()
    mx.random.seed(73)
    gdn = model.text.layers[10].linear_attn
    x = mx.random.normal((1, 32, 1024)).astype(mx.bfloat16)
    cot = mx.random.normal(x.shape)
    results = []
    for backend in ("reference", "native"):
        select(model, backend)
        y = gdn(x)
        _, grad = nn.value_and_grad(gdn, lambda m: (m(x) * cot).sum())(gdn)
        mx.eval(y, grad)
        results.append((y, dict(tree_flatten(grad))))
    error = {k: relative(v, results[0][1][k]) for k, v in results[1][1].items()}
    output = relative(results[1][0], results[0][0])
    if max(error.values()) > 0.035 or output > 0.01:
        raise AssertionError((output, error))
    result = dict(output=output, gradients=error)
    save("outputs/gdn_projection.json", result)
    return result


def masked_contract():
    from mlx_lm.models.gated_delta import gated_delta_ops

    mx.random.seed(7)
    q, k, v = [mx.random.normal((1, 6, 2, 32)) * 0.1 for _ in range(3)]
    decay = mx.full((1, 6, 2), 0.8)
    beta = mx.full((1, 6, 2), 0.3)
    mask = mx.array([[True, False, True, True, False, True]])
    args = (q, k, v, decay, beta, None, mask)
    y, s = gated_delta_ops(*args)
    rows = {}
    for name, stream in (("cpu", mx.cpu), ("metal", mx.gpu)):
        native, state = mx.fast.gated_delta_update(*args, stream=stream)
        valid = mx.where(mask[..., None, None], native - y, 0)
        row = dict(
            valid_max_abs=mx.max(mx.abs(valid)).item(),
            state_relative=relative(state, s),
            masked_output_max_abs=mx.max(
                mx.abs(mx.where(mask[..., None, None], 0, native))
            ).item(),
        )
        if row["valid_max_abs"] > 1e-5 or row["state_relative"] > 1e-5:
            raise AssertionError(row)
        rows[name] = row
    save("outputs/gdn_mask.json", rows)
    return rows
