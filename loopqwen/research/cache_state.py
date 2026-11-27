"""Compare intermediate hidden-output and physical-state materialization at fixed recurrent depth."""

import gc
import hashlib
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from ..evaluation.generation_eval import advance_cache, generate
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..tool_model import bind
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .paired_projection import decision as crossover
from .registered import registered

NAME = "cache-state-prefill-01"


def snapshot(model, ids, lean):
    cache = model.make_cache()
    x = mx.array([ids])
    while x.shape[1] > 128:
        block, x = x[:, :128], x[:, 128:]
        if lean:
            advance_cache(model, block, cache)
        else:
            mx.eval(model.hidden(block, cache))
    logits = model.project(model.hidden(x, cache)[:, -1:])[0, -1]

    def array_sha(v):
        # Preserve the exact bit pattern, including BF16 data and native cache shapes.
        raw = np.asarray(v.view(mx.uint8)).tobytes()
        return [str(v.dtype), list(v.shape), hashlib.sha256(raw).hexdigest()]

    def fingerprint_state(value):
        if isinstance(value, mx.array):
            return array_sha(value)
        if isinstance(value, (tuple, list)):
            return [fingerprint_state(v) for v in value]
        return value

    result = {"logits": array_sha(logits), "sites": []}
    for (layer, visit), slot in cache.slots.items():
        fields = fingerprint_state(slot.state)
        offset = getattr(slot, "offset", None)
        if isinstance(offset, mx.array):
            offset = offset.tolist()
        result["sites"].append(
            {
                "layer": layer,
                "visit": visit,
                "offset": offset,
                "state": fields,
            }
        )
    return result


def decide(rounds, states):
    # Reuse the original balanced-crossover timing accounting. The new study
    # requires bit equality for the complete vocabulary and every physical state.
    blocks = [
        {**r, "mode": "paired" if r["mode"] == "state" else "stock"} for r in rounds
    ]
    d = crossover(blocks, [])
    t = d["totals"]
    t["state"] = t.pop("paired")
    d["gates"]["numeric"] = states["stock"] == states["state"]
    d["gates"]["first_latency"] = (
        t["state"]["first_s"] <= 0.975 * t["stock"]["first_s"]
        and sum(x < 1 for x in d["first_ratios"]) >= 3
    )
    d["passed"] = all(d["gates"].values())
    return d


def verify(root, d):
    spec = registered(NAME)
    assert read(root / "registration.json") == spec
    if d.get("status") == "interrupted":
        assert d["reason"] and not d["passed"]
        return
    rounds = unpack(root / "rounds.json.gz")
    states = read(root / "physical-state-digests.json")
    assert [r["mode"] for r in rounds] == spec["order"]
    assert all([q["id"] for q in r["items"]] == spec["ids"] for r in rounds)
    for panels in states.values():
        assert list(panels) == spec["ids"]
        for v in panels.values():
            assert (
                v["logits"][1][0] >= 200000
                and len(v["sites"]) == 25
                and all(
                    (
                        len(s["state"]) == 3
                        and s["state"][2] == s["offset"]
                        and all(len(tensor) == 3 for tensor in s["state"][:2])
                        if (s["layer"] + 1) % 4 == 0
                        else len(s["state"][0]) == 2 and s["state"][1:] == [None, None]
                    )
                    for s in v["sites"]
                )
            )
    assert d == decide(rounds, states)


def run():
    s = registered(NAME)
    root = Path("outputs/research") / NAME
    root.mkdir(exist_ok=True)
    assert not (root / "decision.json").exists()
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    save(root / "registration.json", s)
    capture(
        root,
        (
            "loopqwen/research/cache_state.py",
            "loopqwen/research/paired_projection.py",
            "loopqwen/evaluation/generation_eval.py",
            "loopqwen/looped_model.py",
            "loopqwen/recurrent_cache.py",
            "scripts/cache_state_sources.py",
        ),
    )
    guard = Budget(NAME, s["limits"])
    try:
        tok = tokenizer(s)
        docs = read(s["material"])["rows"]
        prompts = [tok.decode(w["initial_ids"]) for w in docs]
        assert all(
            tok.encode(v, add_special_tokens=False) == w["initial_ids"]
            for v, w in zip(prompts, docs, strict=True)
        )
        rounds = []
        states = {}
        for i, mode in enumerate(s["order"]):
            guard.check()
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            start = perf_counter()
            model = bind(None, Path(s["checkpoint"]) / "tool")
            mx.synchronize()
            load = perf_counter() - start
            peak = mx.get_peak_memory()
            lean = mode == "state"
            generate(model, tok, s["warmup"], max_tokens=2, prefill_cache_state=lean)
            items = []
            for w, prompt in zip(docs, prompts, strict=True):
                guard.check()
                mx.reset_peak_memory()
                q = generate(
                    model,
                    tok,
                    prompt,
                    max_tokens=s["tokens"],
                    record_kv_history=True,
                    prefill_cache_state=lean,
                )
                items.append(
                    {
                        "id": w["id"],
                        **{
                            k: q[k]
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
            rounds.append(
                {
                    "mode": mode,
                    "load_s": load,
                    "construction_peak": peak,
                    "items": items,
                }
            )
            if i >= len(s["order"]) - 2:
                states[mode] = {
                    w["id"]: snapshot(model, w["initial_ids"], lean) for w in docs
                }
            print("Cache materialization round", i + 1, mode, flush=True)
            del model
        packed(root / "rounds.json.gz", rounds)
        save(root / "physical-state-digests.json", states)
        d = decide(rounds, states)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        d = {"status": "interrupted", "passed": False, "reason": str(exc)}
    finally:
        guard.close()
    save(root / "decision.json", d)
    verify(root, d)


def finish_numeric():
    """Recover only the untimed state probes if serialization failed after all timed rounds."""
    root = Path("outputs/research") / NAME
    s = registered(NAME)
    rounds = unpack(root / "rounds.json.gz")
    assert [r["mode"] for r in rounds] == s["order"] and not (
        root / "decision.json"
    ).exists()
    guard = Budget(NAME, s["limits"])
    try:
        states = {}
        for mode in ("state", "stock"):
            model = bind(None, Path(s["checkpoint"]) / "tool")
            model.eval()
            probes = {}
            for w in read(s["material"])["rows"]:
                guard.check()
                probes[w["id"]] = snapshot(model, w["initial_ids"], mode == "state")
            states[mode] = probes
            del model
            gc.collect()
            mx.clear_cache()
        save(root / "physical-state-digests.json", states)
        save(
            root / "numeric-recovery.json",
            {
                "timed_rounds": digest(root / "rounds.json.gz"),
                "source": digest(Path(__file__)),
                "reason": "The original eight timed rounds completed. Nested MLX convolution and delta tensors needed recursive fingerprinting for the untimed equality probes. The timing blocks were preserved and only the probes were re-executed.",
            },
        )
        d = decide(rounds, states)
        save(root / "decision.json", d)
        verify(root, d)
    finally:
        guard.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finish-numeric", action="store_true")
    (finish_numeric if p.parse_args().finish_numeric else run)()
