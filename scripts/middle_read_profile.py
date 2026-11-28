"""Numerical admission of the single middle-layer read alternative on Apple MLX."""

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, save
from loopqwen.middle_read import LAYER, install, set_reads
from loopqwen.offloaded_prefill import OffloadedPrefill
from loopqwen.shared_query import memory
from loopqwen.tool_cli import open_assistant
from scripts.middle_depth_budget import ROOT, worker
from scripts.middle_depth_material import ROWS
from scripts.middle_depth_standalone import MODELS, check

PLAN = Path("research/diagnostics/middle-depth/middle-read.md")
REPORT = ROOT / "middle-read"
SEED = 7793


def difference(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    delta = a - b
    return {
        "max": mx.max(mx.abs(delta)).item(),
        "relative_rms": mx.sqrt(
            mx.mean(delta**2) / mx.maximum(mx.mean(a**2), 1e-12)
        ).item(),
        "argmax_equal": mx.all(mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)).item(),
    }


def recomputed(layer, x):
    """Use stock Qwen attention and recompute immutable source K/V on each read."""

    class ReadOnly:
        offset = 0

        def __init__(self, k, v):
            self.k, self.v = k, v

        def update_and_fetch(self, k, v):
            return self.k, self.v

    state = layer.block(x, mask="causal")
    for _ in range(2):
        k, v, _, _ = memory(layer.block, x, None, None, mask_needed=False)
        source = layer.block.input_layernorm(state)
        query = (source + layer.step.query(source.astype(mx.float32))).astype(
            source.dtype
        )
        update = layer.block.self_attn(query, mask="causal", cache=ReadOnly(k, v))
        state = layer.step(state, update)
    return state


def profile():
    path = REPORT / "profile.json"
    with worker("profile", "middle-read/profile") as guard:
        root = MODELS / "loop"
        manifest = check(root, "loop")
        assist = open_assistant(argparse.Namespace(checkpoint=root, tool_format="qwen"))
        model = assist._tool_model
        model.span = replace(model.span, passes=1)
        rows = unpack(ROWS)
        row = min(
            (r for r in rows if r["group"] == "open"),
            key=lambda r: len(r["p"]) + len(r["y"]),
        )
        ids = mx.array([row["p"][-181:]])
        parent = model(ids)[:, -1]
        mx.eval(parent)
        layer = install(model, 2, SEED)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if not trainable or any(
            not k.startswith(f"text.model.layers.{LAYER}.step.") for k in trainable
        ):
            raise ValueError(f"Unexpected trainable parameters: {list(trainable)}")
        initial = {str(n): None for n in (1, 2)}
        for n in (1, 2):
            set_reads(model, n)
            initial[str(n)] = difference(parent, model(ids)[:, -1])
        guard()
        x = mx.array([row["p"] + row["y"][:-1]])
        y = mx.array(row["y"])
        model.train()

        def loss():
            hidden = model.hidden(x)[:, -len(row["y"]) :]
            return nn.losses.cross_entropy(
                model.project(hidden)[0].astype(mx.float32), y
            ).mean()

        started = monotonic()
        loss_value, gradients = nn.value_and_grad(layer.step, loss)()
        mx.eval(loss_value, gradients)
        norms = {
            k: mx.linalg.norm(v.astype(mx.float32)).item()
            for k, v in tree_flatten(gradients)
        }
        gradient_seconds = monotonic() - started
        grade = {"initial": initial, "gradient_norms": norms, "loss": loss_value.item()}
        guard()
        model.eval()
        # An explicit nonzero diagnostic write probes the path before any fitting.
        layer.step.bias = mx.full(layer.step.bias.shape, 0.10)
        entry = model._execute(model.text.model.embed_tokens(ids), model.sites[:LAYER])
        expected = recomputed(layer, entry)
        slot = KVCache()
        observed = layer(entry, mask="causal", cache=slot)
        grade["stock_recomputed_memory"] = difference(expected, observed)
        grade["middle_offset"] = slot.offset
        grade["middle_retained"] = slot.keys_and_values()[0].shape[2]
        set_reads(model, 1)
        one = model(ids)[:, -1]
        cache = model.make_cache()
        set_reads(model, 2)
        try:
            cache.validate(model, None)
        except ValueError:
            invalidated = True
        else:
            invalidated = False
        full = model(ids)
        grade["later_read_influence"] = difference(one, full[:, -1])
        altered = mx.concatenate([ids[:, :128], mx.zeros_like(ids[:, 128:])], axis=1)
        grade["causal_prefix"] = difference(full[:, :128], model(altered)[:, :128])
        cache = model.make_cache()
        h = model.hidden(ids[:, :128], cache=cache)
        mx.eval(h)
        rest = model.hidden(ids[:, 128:], cache=cache)
        cached = model.project(rest[:, -1])
        grade["chunked"] = difference(full[:, -1], cached)
        grade["invalidated_on_depth_change"] = invalidated
        grade["attention_offsets"] = {
            str(i): cache.slots[i, 0].offset
            for i in range(24)
            if not model.text.layers[i].is_linear
        }
        grade["hybrid_histories"] = sum(
            model.text.layers[i].is_linear
            and all(t is not None for t in cache.slots[i, 0].state[0])
            for i in range(24)
        )
        guard()
        with TemporaryDirectory() as directory:
            disk = OffloadedPrefill(model, directory, group_attention=True)
            staged = disk.prefill(ids[0, :128].tolist(), progress=lambda _: guard())
            grade["staged_prefix"] = difference(
                model.project(h[:, -1]), staged.reshape(1, -1)
            )
            grade["staged_next"] = difference(cached, disk.decode(ids[:, 128:])[:, -1])
            scratch = disk.scratch_bytes
        import math

        passed = (
            all(r["max"] == 0 for r in initial.values())
            and math.isfinite(grade["loss"])
            and all(map(math.isfinite, norms.values()))
            and max(norms.values()) > 0
            and grade["stock_recomputed_memory"]["max"] == 0
            and grade["causal_prefix"]["max"] == 0
            and grade["later_read_influence"]["max"] > 0
            and grade["middle_offset"] == grade["middle_retained"] == ids.shape[1]
            and grade["invalidated_on_depth_change"]
            and grade["hybrid_histories"] == 18
            and len(grade["attention_offsets"]) == 6
            and set(grade["attention_offsets"].values()) == {ids.shape[1]}
            and all(
                grade[k]["relative_rms"] <= 0.05 and grade[k]["argmax_equal"]
                for k in ("chunked", "staged_prefix", "staged_next")
            )
        )
        save(
            path,
            {
                "passed": passed,
                "plan": digest(PLAN),
                "base": manifest["files"]["model.safetensors"],
                "parent": manifest["files"]["adapters.safetensors"],
                "seed": SEED,
                "training": digest(ROWS),
                "source": row["id"],
                "opening_probe": fingerprint(ids.tolist()),
                "probe_positions": ids.shape[1],
                "gradient_positions": x.shape[1],
                "gradient_seconds": gradient_seconds,
                "controller_parameters": sum(v.size for v in trainable.values()),
                "ordinary_layer_calls": 24,
                "extra_attention_reads": [1, 2],
                "scratch_bytes": scratch,
                "peak_mlx": mx.get_peak_memory(),
                **grade,
            },
        )
        if not passed:
            raise ValueError(
                "Middle read failed numerical admission; see the individual checks"
            )


if __name__ == "__main__":
    profile()
