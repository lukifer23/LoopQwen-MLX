"""Native acceptance checks using historical and newly trained adapters."""

import gc
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .. import checkpoint
from ..data.v2 import rows
from ..io import read, save
from ..training.optimizers import optimizer


def relative(actual, reference):
    a, b = actual.astype(mx.float32), reference.astype(mx.float32)
    return mx.sqrt(mx.mean((a - b) ** 2) / mx.mean(b**2)).item()


def run():
    root = Path("checkpoints/p3_learn1")
    generation = checkpoint.location(root)
    state = read(generation / "state.json")
    model, _, _ = checkpoint.setup(state["config"])
    opt = optimizer(state["config"])
    checkpoint.restore(generation, model, opt)
    stored = mx.load(str(generation / "optimizer.safetensors"))
    restored = dict(tree_flatten(opt.state))
    if stored.keys() != restored.keys() or not all(
        mx.array_equal(w, restored[k]).item() for k, w in stored.items()
    ):
        raise AssertionError("Optimizer restoration differs")
    groups = [o.step.item() for o in opt.optimizers]
    if groups != [state["step"]] * 2:
        raise AssertionError(groups)
    examples = rows("validation", "trace")[:3]
    ids = [r["ids"][: r["prefix"]] for r in examples]
    seq = mx.array([ids[0]])
    model.eval()
    original = model.span
    model.span = replace(original, passes=1)
    k1 = model.hidden(seq)
    if len(model.sites) != 24:
        raise AssertionError("Bypass executed extra sites")
    model.span = original
    model.alpha_override = 0.0
    zero = model.hidden(seq)
    if len(model.sites) != 30 or not mx.array_equal(k1, zero).item():
        raise AssertionError("Functional zero control or bypass parity failed")
    model.alpha_override = None
    traces = []
    observed = model.hidden(seq, observer=lambda event, p, h: traces.append((event, p)))
    if not mx.array_equal(observed, model.hidden(seq)).item() or len(traces) != 4:
        raise AssertionError("Observer changed the gated forward")
    cache = model.make_cache()
    if len({id(x) for x in cache.slots.values()}) != 30:
        raise AssertionError("Cache alias")
    full = model(seq)
    chunks = [model(seq[:, i : i + 5], cache=cache) for i in range(0, seq.shape[1], 5)]
    error = relative(mx.concatenate(chunks, axis=1), full)
    pads = [max(map(len, ids)) - len(x) for x in ids]
    c = model.make_cache(left_padding=pads)
    batched = model.project(
        model.hidden(mx.array([[0] * p + x for p, x in zip(pads, ids)]), c)[:, -1]
    )
    c.finish_left_prefill()
    refs = [model.make_cache() for _ in ids]
    serial = mx.concatenate(
        [
            model.project(model.hidden(mx.array([x]), cc)[:, -1])
            for x, cc in zip(ids, refs)
        ]
    )
    padded = [relative(batched, serial)]
    for _ in range(3):
        x = mx.argmax(serial, axis=-1)[:, None]
        batched = model(x, c)[:, 0]
        serial = mx.concatenate(
            [model(x[i : i + 1], cc)[:, 0] for i, cc in enumerate(refs)]
        )
        padded.append(relative(batched, serial))
    if max(error, *padded) > 0.03:
        raise AssertionError((error, padded))
    del model, opt, stored, restored, full, chunks, k1, zero, cache, c, refs
    gc.collect()
    mx.clear_cache()
    old, _, _ = checkpoint.load_trained(Path("checkpoints/p2_confirm_43_loop"))
    old.span = replace(old.span, passes=1)
    err = relative(old(seq), old.text(seq))
    if err != 0:
        raise AssertionError("Historical K=1 no longer matches stock")
    save(
        "outputs/phase3/native-audit.json",
        dict(
            gate_group_steps=groups,
            gate_delta=state["history"][-1]["gate_delta"],
            learned_alpha=state["history"][-1]["alpha"],
            no_op_exact=True,
            observer_events=traces,
            cached_full_relative_rms=error,
            padded_relative_rms=padded,
            historical_k1_relative_rms=err,
        ),
    )


if __name__ == "__main__":
    run()
