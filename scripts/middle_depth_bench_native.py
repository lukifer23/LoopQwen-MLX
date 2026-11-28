"""Score frozen, label-free standard task requests with one installed MLX model."""

import argparse
from dataclasses import replace
from math import isfinite
import os
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.tool_cli import open_assistant


def measure(args):
    tick = monotonic()
    assistant = open_assistant(
        argparse.Namespace(checkpoint=args.checkpoint, tool_format="qwen")
    )
    model, tokenizer = assistant._tool_model, assistant.tokenizer
    if args.one_pass:
        if (
            read(args.checkpoint / "standalone-middle.json")["arm"] != "loop"
            or model.span.passes != 2
        ):
            raise ValueError(
                "The intervention requires the unchanged deployed two-pass loop"
            )
        model.span = replace(model.span, passes=1)
        if len(model.sites) != 24:
            raise ValueError("Physical one-pass layer count differs")
    mx.eval(model.parameters())
    loaded = monotonic() - tick
    policy = args.checkpoint / "registration.json"
    weights = (
        digest(policy.parent / "adapters.safetensors")
        if policy.exists()
        else digest(args.checkpoint / "model.safetensors")
    )
    rows = read(args.requests)
    fixed = read(args.tokens)
    if len(rows) != 250 or len(fixed) != len(rows):
        raise ValueError("Incomplete frozen suite")
    for item in rows:
        task, idx = item["key"].split("/")
        path = args.output / (task + ".json")
        prior = read(path) if path.exists() else {}
        if idx in prior:
            saved = prior[idx]
            if saved["sites"] != len(model.sites):
                raise ValueError("Cannot resume a different pass count")
            if (saved["weights"], saved["request_sha"], saved["token_sha"]) != (
                weights,
                item["sha"],
                fixed[item["key"]],
            ):
                raise ValueError("Cannot resume a different standard benchmark")
            continue
        encoded = []
        for context, continuation in item["requests"]:
            clean = context.rstrip()
            whole = tokenizer.encode(context + continuation, add_special_tokens=False)
            prefix = tokenizer.encode(clean, add_special_tokens=False)
            if (
                not prefix
                or whole[: len(prefix)] != prefix
                or len(prefix) >= len(whole)
            ):
                raise ValueError("Ambiguous tokenizer split")
            encoded.append((prefix, whole[len(prefix) :]))
        token_sha = fingerprint(encoded)
        if token_sha != fixed[item["key"]]:
            raise ValueError("Installed tokenizer differs from frozen inputs")
        choices = []
        peak = 0
        mx.reset_peak_memory()
        reference_seconds = 0.0
        start = monotonic()
        for ctx, target in encoded:
            ids = mx.array([ctx + target[:-1]])
            h = model.hidden(ids)[:, -len(target) :]
            logits = model.project(h).astype(mx.float32)
            t = mx.array(target)
            picked = logits[0, mx.arange(len(target)), t]
            lp = picked - mx.logsumexp(logits[0], axis=-1)
            probs = lp.tolist()
            if not all(isfinite(x) and x <= 0 for x in probs):
                raise ValueError("Invalid conditional log probability")
            if not prior:
                reference_start = monotonic()
                import numpy as np

                # Independent CPU reduction of the actual native logits.
                z = np.asarray(logits[0]).astype(np.float64)
                maximum = z.max(axis=1)
                expected = (
                    z[np.arange(len(target)), target]
                    - maximum
                    - np.log(np.exp(z - maximum[:, None]).sum(axis=1))
                )
                error = max(abs(a - b) for a, b in zip(expected, probs, strict=True))
                if (
                    error > 0.00005
                    or float(
                        mx.max(
                            mx.abs(lp + nn.losses.cross_entropy(logits[0], t))
                        ).item()
                    )
                    > 0.00005
                ):
                    raise ValueError("Native score failed independent reference")
                reference_seconds += monotonic() - reference_start
            else:
                error = None
            peak = max(peak, mx.get_peak_memory())
            choices.append(
                {
                    "context_tokens": len(ctx),
                    "target_tokens": len(target),
                    "token_logp": probs,
                    "cpu_reference_max_error": float(error)
                    if error is not None
                    else None,
                }
            )
        prior[idx] = {
            "weights": weights,
            "request_sha": item["sha"],
            "token_sha": token_sha,
            "choices": choices,
            "seconds": monotonic() - start - reference_seconds,
            "reference_seconds": reference_seconds,
            "peak_mlx": peak,
            "pid": os.getpid(),
            "sites": len(model.sites),
        }
        save(path, prior)
    save(
        args.output / "loaded.json",
        {
            "weights": weights,
            "seconds": loaded,
            "pid": os.getpid(),
            "loaded_once_per_process": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "requests", "tokens", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--one-pass", action="store_true")
    measure(p.parse_args())
