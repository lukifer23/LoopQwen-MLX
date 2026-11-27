"""Reload the sealed four-layer checkpoint and independently reproduce all 32 native probe losses."""

from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.io import digest, read, save
from loopqwen.reentry import resident
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_reentry_fit import NAME, inputs
from loopqwen.worker import acquire


def losses(net, state, offsets, gold):
    logits = net.project(net.text.model.norm(state[0, offsets])).astype(mx.float32)
    selected = mx.take_along_axis(logits, gold[:, None], axis=-1)[:, 0]
    return (mx.logsumexp(logits, axis=-1) - selected).mean().item()


def replay():
    root = Path("research/results") / NAME
    s = read(root / "registration.json")
    fit = read(root / "fit.json")
    decision = read(root / "decision.json")
    files = read(root / "checksums.json")
    for name in (
        "registration.json",
        "fit.json",
        "decision.json",
        "training-probes.json",
        "reentry.safetensors",
    ):
        assert digest(root / name) == files[name]
    assert fit["weights"] == files["reentry.safetensors"]
    acquire()
    net = resident(s["checkpoint"], root, s, fit["backbone"])
    weights = mx.load(str(root / "reentry.safetensors"))
    expected = dict(tree_flatten(net.trainable_parameters()))
    assert weights.keys() == expected.keys() and all(
        t.shape == expected[k].shape and t.dtype == expected[k].dtype
        for k, t in weights.items()
    )
    net.load_weights(list(weights.items()), strict=False)
    net.eval()
    rows = inputs(tokenizer(s), s)
    results = []
    for measured in read(root / "training-probes.json"):
        key = measured["view"]
        _, prefix, y, _ = rows[key]
        first = net.first_visit(mx.array([prefix + y[:-1]]))
        second = net.second_visit(first)
        offsets = mx.array(measured["positions"]) + len(prefix) - 1
        gold = mx.array([y[i] for i in measured["positions"]])
        a, b = (losses(net, h, offsets, gold) for h in (first, second))
        results.append(
            {
                "view": key,
                "positions": len(measured["positions"]),
                "first_nll": a,
                "second_nll": b,
                "first_abs_error": abs(a - measured["first_nll"]),
                "second_abs_error": abs(b - measured["second_nll"]),
            }
        )
    assert len(results) == 32
    err = max(max(r["first_abs_error"], r["second_abs_error"]) for r in results)
    gain = mean(r["first_nll"] - r["second_nll"] for r in results)
    assert err <= 0.0001 and abs(gain - decision["training_mean_nll_gain"]) <= 0.0001
    assert (gain >= 0.05) == decision["gates"]["probe_gain"]
    receipt = {
        "registration": files["registration.json"],
        "weights": fit["weights"],
        "backbone": fit["backbone"]["sha256"],
        "measured_archive": digest(root / "checksums.json"),
        "positions": sum(r["positions"] for r in results),
        "max_abs_nll_error": err,
        "independent_mean_nll_gain": gain,
        "views": results,
    }
    dest = Path("docs/measurements/table-reentry-fit-native-replay.json")
    if dest.is_file():
        assert read(dest) == receipt
    else:
        save(dest, receipt)
    print(
        len(results),
        "reloaded training turns",
        receipt["positions"],
        "target positions",
        "mean gain",
        gain,
        "maximum absolute error",
        err,
        flush=True,
    )


if __name__ == "__main__":
    replay()
