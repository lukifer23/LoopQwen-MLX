"""Measure whether joining immutable SwiGLU inputs preserves the released recurrent function."""

import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from ..evaluation.generation_eval import generate
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..kernels.paired_projection import pair_frozen_prefix
from ..tool_model import bind
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .registered import registered

NAME = "paired-projection-01"


def decision(rounds, check):
    totals = {}
    for mode in ("stock", "paired"):
        runs = [r for r in rounds if r["mode"] == mode]
        flat = [x for r in runs for x in r["items"]]
        first = sum(x["ttft_s"] for x in flat)
        elapsed = sum(x["elapsed_s"] for x in flat)
        totals[mode] = {
            "first_s": first,
            "elapsed_s": elapsed,
            "peak_bytes": max(x["peak_bytes"] for x in flat),
            "load_s": sum(r["load_s"] for r in runs),
            "construction_peak": max(r["construction_peak"] for r in runs),
            "decode_tps": sum(len(x["token_ids"]) - 1 for x in flat)
            / (elapsed - first),
            "input_tps": sum(x["prompt_tokens"] for x in flat) / first,
        }
    stock, fused = totals["stock"], totals["paired"]
    comparisons = []
    identical = True
    for i in range(0, len(rounds), 2):
        pair = {r["mode"]: r for r in rounds[i : i + 2]}
        left, right = pair["stock"]["items"], pair["paired"]["items"]
        for l, r in zip(left, right, strict=True):
            assert (l["id"], l["prompt_tokens"]) == (r["id"], r["prompt_tokens"])
            identical &= (
                l["token_ids"] == r["token_ids"] and l["kv_history"] == r["kv_history"]
            )
        comparisons.append(
            sum(w["ttft_s"] for w in right) / sum(w["ttft_s"] for w in left)
        )
    gates = {
        "identical_tokens_and_histories": bool(identical),
        "numeric": all(
            x["same_argmax"] and x["max_abs"] <= 0.125 and x["rms"] <= 0.02
            for x in check
        ),
        "first_latency": fused["first_s"] <= 0.95 * stock["first_s"]
        and sum(x < 1 for x in comparisons) >= 3,
        "decode": fused["decode_tps"] >= stock["decode_tps"],
        "peak": fused["peak_bytes"] <= stock["peak_bytes"],
        "load": fused["load_s"] <= 1.05 * stock["load_s"],
        "construction_peak": fused["construction_peak"] <= stock["construction_peak"],
    }
    return {
        "passed": all(gates.values()),
        "totals": totals,
        "first_ratios": comparisons,
        "gates": gates,
    }


def read_numeric(root, ids):
    result = []
    with (
        safe_open(root / "numeric-stock.safetensors", framework="np") as stock,
        safe_open(root / "numeric-paired.safetensors", framework="np") as fused,
    ):
        assert set(stock.keys()) == set(fused.keys()) == set(ids)
        for ident in ids:
            left, right = stock.get_tensor(ident), fused.get_tensor(ident)
            assert left.shape == right.shape and left.ndim == 1 and left.size >= 200000
            delta = right.astype(np.float64) - left.astype(np.float64)
            result.append(
                {
                    "id": ident,
                    "same_argmax": int(np.argmax(left)) == int(np.argmax(right)),
                    "max_abs": float(np.max(np.abs(delta))),
                    "rms": float(np.sqrt(np.mean(delta**2))),
                }
            )
    return result


def verify(root, d):
    s = registered(NAME)
    assert s == read(root / "registration.json")
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    records = unpack(root / "rounds.json.gz")
    pairs = read_numeric(root, s["ids"])
    assert pairs == read(root / "numeric.json")
    assert len(records) == 8 and len(pairs) == len(s["ids"])
    assert [r["mode"] for r in records] == s["order"]
    assert all([r["id"] for r in block["items"]] == s["ids"] for block in records)
    assert [r["id"] for r in pairs] == s["ids"] and d == decision(records, pairs)


def run():
    s = registered(NAME)
    out = Path("outputs/research") / NAME
    out.mkdir(exist_ok=True)
    assert not (out / "decision.json").exists()
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    save(out / "registration.json", s)
    capture(
        out,
        (
            "loopqwen/research/paired_projection.py",
            "loopqwen/kernels/paired_projection.py",
            "loopqwen/evaluation/generation_eval.py",
            "loopqwen/tool_model.py",
            "loopqwen/looped_model.py",
        ),
    )
    guard = Budget(NAME, s["limits"])
    try:
        tok = tokenizer(s)
        panel = read(s["material"])
        indexed = {w["id"]: w for w in panel["rows"]}
        seq = [indexed[k] for k in s["ids"]]
        prompts = [tok.decode(w["initial_ids"]) for w in seq]
        assert all(
            tok.encode(p, add_special_tokens=False) == w["initial_ids"]
            for p, w in zip(prompts, seq, strict=True)
        )
        blocks = []
        for number, mode in enumerate(s["order"]):
            guard.check()
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            start = perf_counter()
            model = bind(None, Path(s["checkpoint"]) / "tool")
            if mode == "paired":
                assert pair_frozen_prefix(model) == 23
            mx.synchronize()
            loaded = perf_counter() - start
            construction = mx.get_peak_memory()
            generate(model, tok, s["warmup"], max_tokens=2)
            traces = []
            for q, p in zip(seq, prompts, strict=True):
                guard.check()
                mx.reset_peak_memory()
                output = generate(
                    model, tok, p, max_tokens=s["tokens"], record_kv_history=True
                )
                traces.append(
                    {
                        "id": q["id"],
                        **{
                            k: output[k]
                            for k in (
                                "ttft_s",
                                "elapsed_s",
                                "peak_bytes",
                                "prompt_tokens",
                                "token_ids",
                                "kv_history",
                            )
                        },
                    }
                )
            blocks.append(
                {
                    "mode": mode,
                    "load_s": loaded,
                    "construction_peak": construction,
                    "items": traces,
                }
            )
            print("Projection crossover", number + 1, mode, flush=True)
            del model
        for mode in ("stock", "paired"):
            guard.check()
            gc.collect()
            mx.clear_cache()
            model = bind(None, Path(s["checkpoint"]) / "tool")
            if mode == "paired":
                assert pair_frozen_prefix(model) == 23
            values = {}
            for item in seq:
                guard.check()
                logits = model(mx.array([item["initial_ids"]]))[0, -1].astype(
                    mx.float32
                )
                mx.eval(logits)
                values[item["id"]] = logits
            mx.save_safetensors(str(out / ("numeric-" + mode + ".safetensors")), values)
            del logits, values, model
        numeric = read_numeric(out, s["ids"])
        packed(out / "rounds.json.gz", blocks)
        save(out / "numeric.json", numeric)
        d = decision(blocks, numeric)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        d = {"status": "interrupted", "passed": False, "reason": str(exc)}
    finally:
        guard.close()
    save(out / "decision.json", d)
    verify(out, d)


if __name__ == "__main__":
    run()
