"""A bounded-working-set native adjoint for the complete causal middle-span objective."""

import argparse
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import architecture, donor
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_fit import RUN, SPEC, feature, loss
from scripts.middle_depth_material import ROWS


def bounded(model, anchor, labels, start, depth, negatives, guard):
    # Store native training-forward boundary tensors only. The reverse pass
    # recomputes one complete causal layer and adds each tied visit's cotangent.
    activations = []
    h = anchor
    gradients = {}
    weights = dict(tree_flatten(model.trainable_parameters()))

    def layer_call(index):
        layer = model.text.layers[index]
        items = tree_flatten(layer.trainable_parameters())
        names = [k for k, _ in items]
        values = [v for _, v in items]

        def execute(x, *params):
            if names:
                layer.update(tree_unflatten(list(zip(names, params, strict=True))))
            return model._execute(x, [(index, 0)])

        full_names = ["text.model.layers." + str(index) + "." + k for k in names]
        return execute, values, full_names

    def mix(x, gate):
        return (x + mx.sigmoid(gate) * (anchor - x)).astype(x.dtype)

    def apply(fn, values, names):
        nonlocal h
        guard()
        inputs = [h, *values]
        # SDPA can choose a different forward kernel inside a VJP. Use its
        # differentiated forward even though we discard this zero cotangent.
        outputs, _ = mx.vjp(fn, inputs, [mx.zeros_like(h)])
        activations.append((fn, inputs, names))
        h = outputs[0]
        mx.eval(h)

    for visit in range(depth):
        if visit:
            apply(mix, [weights["alpha_logit"]], ["alpha_logit"])
        for i in range(12, 16):
            apply(*layer_call(i))
    for i in range(16, 24):
        apply(*layer_call(i))

    def objective(x):
        hidden = model.text.model.norm(x)[:, start:]
        logits = model.project(hidden)[0].astype(mx.float32)
        value = nn.losses.cross_entropy(logits, labels).mean()
        if negatives:
            pos, token = zip(*negatives)
            ix = mx.array(pos)
            value += (
                0.65
                * nn.softplus(
                    1 + logits[ix, mx.array(token)] - logits[ix, labels[ix]]
                ).mean()
            )
        return value

    value, adjoint = mx.value_and_grad(objective)(h)
    mx.eval(value, adjoint)
    while activations:
        guard()
        fn, inputs, names = activations.pop()
        _, backward = mx.vjp(fn, inputs, [adjoint])
        adjoint = backward[0]
        for k, g in zip(names, backward[1:], strict=True):
            gradients[k] = gradients[k] + g if k in gradients else g
        mx.eval(adjoint, gradients)
    # Restore parameter references after MLX traced module.update. Return the
    # identical optimizer tree, including a zero one-pass mixing derivative.
    model.update(tree_unflatten(list(weights.items())))
    result = tree_unflatten(
        [
            (k, gradients[k] if k in gradients else mx.zeros_like(v))
            for k, v in weights.items()
        ]
    )
    return value, result


def compare(arm):
    spec = read(SPEC)
    rows = unpack(ROWS)
    first = next(r for r in rows if r["group"] == "polarity")
    second = next(
        r
        for r in rows
        if r["group"] in ("conversation", "deliver")
        and 130 <= len(r["p"]) + len(r["y"]) <= 360
    )
    result = []
    with worker("profile", "layerwise-adjoint/" + arm) as guard:
        text, _ = donor(BASE, "base")
        trained = 3 if arm == "loop" else 1
        model = architecture(text, trained, spec["seed"])
        path = RUN / str(spec["seed"]) / arm / "step-0144"
        record = read(path / "state.json")
        file = path / "adapters.safetensors"
        assert digest(file) == record["hashes"]["adapters.safetensors"]
        model.load_weights(str(file), strict=False)
        model.train()
        reference = nn.value_and_grad(model, loss)
        for r in (first, second):
            anchor, _ = feature(model, r, record["identity"]["feature_id"])
            for k in range(1, trained + 1):
                args = (
                    model,
                    anchor,
                    mx.array(r["y"]),
                    len(r["p"]) - 1,
                    k,
                    r["negatives"],
                )
                guard()
                mx.clear_cache()
                mx.reset_peak_memory()
                tick = monotonic()
                expected, g = reference(*args)
                mx.eval(expected, g)
                original_peak = mx.get_peak_memory()
                original_seconds = monotonic() - tick
                mx.clear_cache()
                mx.reset_peak_memory()
                tick = monotonic()
                actual, h = bounded(*args, guard)
                mx.eval(actual, h)
                peak = mx.get_peak_memory()
                a, b = dict(tree_flatten(g)), dict(tree_flatten(h))
                assert a.keys() == b.keys()
                norms = {}
                for key in a:
                    x, y = a[key].astype(mx.float32), b[key].astype(mx.float32)
                    norms[key] = [
                        mx.sum(z).item() for z in (x * x, y * y, x * y, (x - y) ** 2)
                    ]
                xx, yy, xy, ee = map(sum, zip(*norms.values(), strict=True))
                rel = (ee / xx) ** 0.5
                cos = xy / (xx * yy) ** 0.5
                err = abs(actual.item() - expected.item())
                result.append(
                    dict(
                        source=r["id"],
                        positions=len(r["p"]) + len(r["y"]) - 1,
                        depth=k,
                        loss=[expected.item(), actual.item()],
                        cosine=cos,
                        relative=rel,
                        tensors=norms,
                        reference_peak=original_peak,
                        bounded_peak=peak,
                        reference_seconds=original_seconds,
                        bounded_seconds=monotonic() - tick,
                        passed=err < 0.001 and rel < 0.01 and cos > 0.9999,
                    )
                )
                save(
                    ROOT / f"adjoint-{arm}.json",
                    dict(
                        implementation=digest(Path(__file__)),
                        policy=digest(file),
                        rows=result,
                        passed=len(result) == 2 * trained
                        and all(s["passed"] for s in result),
                    ),
                )
                print(arm, k, r["group"], err, rel, cos, flush=True)
                if not result[-1]["passed"]:
                    raise ValueError("Native training adjoints differ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("ordinary", "loop"))
    compare(p.parse_args().arm)
