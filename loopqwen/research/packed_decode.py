"""Compare native single-query attention implementations on physical Qwen KV rows."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from ..deployment import load_trained
from ..identity import fingerprint
from ..io import digest, read, save
from ..kernels.packed_decode import attention
from ..page_prefill import DirectPackedPageCache
from .core import Budget
from .registered import registered


def tensor_sha(x):
    a = np.asarray(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_case(q, cache, section, spec, outputs, events, budget):
    from mlx_lm.models.base import quantized_scaled_dot_product_attention as stock

    length = cache.offset
    bundles = cache.keys_and_values()
    width = q.shape[2]
    for i in range(width):
        n = length - width + i + 1
        query = q[:, :, i : i + 1]
        k, v = [[a[:, :, :n] for a in bundle] for bundle in bundles]
        pos = mx.arange(n)
        reader_bias = mx.where(
            (pos >= section["start"]) & (pos < section["stop"]), section["gain"], 0.0
        ).astype(mx.bfloat16)[None, None, None, :]
        for flavor, mask in (("plain", None), ("reader", reader_bias)):
            inputs = [query, *k, *v] + ([] if mask is None else [mask])
            mx.eval(inputs)
            # Compile/prime each exact shape independently, outside the timed repeats.
            for call in (stock, attention):
                mx.eval(call(mx.array(query), k, v, 0.0625, mask))
            for rep in range(spec["repeats"]):
                order = (
                    ("stock", "fused")
                    if (i + rep + (flavor == "reader")) % 2 == 0
                    else ("fused", "stock")
                )
                for side in order:
                    mx.synchronize()
                    mx.reset_peak_memory()
                    active = mx.get_active_memory()
                    t = perf_counter()
                    y = (attention if side == "fused" else stock)(
                        mx.array(query), k, v, 0.0625, mask
                    )
                    mx.eval(y)
                    elapsed = perf_counter() - t
                    events.append(
                        {
                            "side": side,
                            "length": length,
                            "prefix": n,
                            "mask": flavor,
                            "rep": rep,
                            "seconds": elapsed,
                            "extra_bytes": mx.get_peak_memory() - active,
                        }
                    )
                    if rep == 0:
                        outputs[f"{side}-{n}-{flavor}"] = mx.array(
                            np.asarray(y.astype(mx.float32))
                        )
                    del y
                budget.check()
    return {
        "length": length,
        "retained": cache.physical_length(),
        "capacity": cache.capacity,
        "query_sha": tensor_sha(q),
        "kv_sha": [tensor_sha(x) for bundle in bundles for x in bundle],
    }


def outcome(spec, events, values):
    from statistics import median

    groups = {}
    max_error = 0.0
    minimum_cosine = 1.0
    for length in spec["lengths"]:
        for mask in ("plain", "reader"):
            key = f"{length}-{mask}"
            paired = {}
            peak = {}
            for side in ("stock", "fused"):
                rows = [
                    e
                    for e in events
                    if e["length"] == length and e["mask"] == mask and e["side"] == side
                ]
                identities = {(r["prefix"], r["rep"]) for r in rows}
                assert len(rows) == len(identities) == spec["queries"] * spec["repeats"]
                assert identities == {
                    (n, i)
                    for n in range(length - spec["queries"] + 1, length + 1)
                    for i in range(spec["repeats"])
                }
                assert all(r["seconds"] > 0 and r["extra_bytes"] >= 0 for r in rows)
                paired[side] = median(r["seconds"] for r in rows)
                peak[side] = max(r["extra_bytes"] for r in rows)
            groups[key] = {
                "median_seconds": paired,
                "extra_peak_bytes": peak,
                "speedup": paired["stock"] / paired["fused"],
            }
            for n in range(length - spec["queries"] + 1, length + 1):
                a, b = (
                    np.asarray(values[f"{s}-{n}-{mask}"], dtype=np.float32).ravel()
                    for s in ("fused", "stock")
                )
                assert (
                    a.shape == b.shape == (8 * 256,)
                    and np.isfinite(a).all()
                    and np.isfinite(b).all()
                )
                error = float(
                    np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b))))
                )
                cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                max_error = max(error, max_error)
                minimum_cosine = min(cosine, minimum_cosine)
    passed = (
        max_error <= spec["max_error"]
        and minimum_cosine >= spec["min_cosine"]
        and all(
            v["speedup"] >= spec["min_speedup"]
            for k, v in groups.items()
            if not k.startswith(str(spec["lengths"][0]) + "-")
        )
    )
    return {
        "passed": bool(passed),
        "max_scaled_abs_error": max_error,
        "min_cosine": minimum_cosine,
        "groups": groups,
    }


def verify(root, d):
    import safetensors.numpy as st

    spec = read(root / "registration.json")
    assert digest(Path(spec["parent"]) / "checksums.json") == spec["parent_sha256"]
    receipts = read(root / "cache.json")
    assert [r["length"] for r in receipts] == spec["lengths"]
    assert all(
        r["retained"] == r["length"]
        and r["capacity"] == spec["lengths"][-1] + 95
        and len(r["kv_sha"]) == 6
        for r in receipts
    )
    assert (
        outcome(
            spec,
            read(root / "events.json"),
            st.load_file(str(root / "outputs.safetensors")),
        )
        == d
    )


def run(name):
    from ..evaluation.mechanism import unpack

    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    parent = Path(spec["parent"])
    if (
        digest(parent / "checksums.json") != spec["parent_sha256"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Unqualified physical input")
    row = read(spec["material"])[0]["panel"][0]
    ids = row["prompt_ids"]
    ref = unpack(parent / "direct_pages.json.gz")[0]
    if fingerprint(ids) != ref["input_sha256"]:
        raise ValueError("Different 131K input")
    budget = Budget(name, spec["limits"])
    try:
        model, _, _ = load_trained(spec["checkpoint"])
        state = model.make_cache()
        layer = model.text.layers[3]
        attn = layer.self_attn
        final = spec["lengths"][-1]
        cache = DirectPackedPageCache(final, final + 95)
        events = []
        values = {}
        receipts = []
        for length in spec["lengths"]:
            width = spec["queries"]
            while cache.offset < length:
                a = cache.offset
                b = min(a + 128, length - width) if a < length - width else length
                tokens = mx.array([ids[a:b]])
                h = model._execute(
                    model.text.model.embed_tokens(tokens),
                    [(0, 0), (1, 0), (2, 0)],
                    state,
                )
                x = layer.input_layernorm(h)
                n = b - a
                q, _ = mx.split(attn.q_proj(x).reshape(1, n, 8, -1), 2, axis=-1)
                q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=a)
                k = attn.rope(
                    attn.k_norm(attn.k_proj(x).reshape(1, n, 2, 256)).transpose(
                        0, 2, 1, 3
                    ),
                    offset=a,
                )
                v = attn.v_proj(x).reshape(1, n, 2, 256).transpose(0, 2, 1, 3)
                bundles = cache.update_and_fetch(k, v)
                mx.eval(q, bundles)
                budget.check()
            receipts.append(
                run_case(
                    q, cache, ref["attention_memory"], spec, values, events, budget
                )
            )
            print("native operator prefix", length, flush=True)
        save(root / "cache.json", receipts)
        save(root / "events.json", events)
        mx.save_safetensors(str(root / "outputs.safetensors"), values)
        save(root / "decision.json", outcome(spec, events, values))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
