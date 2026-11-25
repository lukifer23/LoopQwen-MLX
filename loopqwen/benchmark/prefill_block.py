"""Balanced native MLX prefill timing and full-vocabulary numerical comparison."""

import gc
from pathlib import Path
from statistics import median

import mlx.core as mx
import numpy as np

from ..deployment import load_trained
from ..evaluation.generation_eval import generate
from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..research import core

REG = Path("research/experiments/prefill-block-01.json")
SOURCE = Path("research/sessions/2026-11-25/context-128k-01/prompt.json.gz")


class VocabularyCapture:
    def __init__(self, model):
        self.model = model
        self.outputs = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def project(self, h):
        logits = self.model.project(h)
        self.outputs.append(logits)
        return logits


def summarize(rows, s):
    result = {}
    for length in s["lengths"]:
        block = {}
        for mode in ("baseline", "treatment"):
            group = [
                r
                for r in rows
                if r["prompt_tokens"] == length and r["block"] == s[mode]
            ]
            if len(group) != 2:
                raise ValueError("Incomplete timing crossover")
            block[mode] = {
                "ttft": median(r["ttft_s"] for r in group),
                "peak_bytes": max(r["peak_bytes"] for r in group),
            }
        group = [r for r in rows if r["prompt_tokens"] == length]
        a, b = block["baseline"], block["treatment"]
        block["identical"] = all(r["token_ids"] == group[0]["token_ids"] for r in group)
        block["max_rms"] = max(max(r["relative_rms"]) for r in group)
        block["passed"] = (
            block["identical"]
            and block["max_rms"] <= 0.005
            and b["ttft"] <= 0.90 * a["ttft"]
            and b["peak_bytes"] <= a["peak_bytes"] + 0.30 * 2**30
        )
        result[str(length)] = block
    return result


def run():
    s = read(REG)
    if digest(Path(s["candidate"]) / "adapters.safetensors") != s["weights"]:
        raise ValueError("Wrong performance checkpoint")
    frozen = unpack(SOURCE)
    ids = frozen["prompt_ids"]
    if fingerprint(ids) != frozen["input_sha256"]:
        raise ValueError("Frozen prompt identity differs")
    out = core.OUT / s["id"]
    if out.exists():
        raise FileExistsError(out)
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    rows = []
    result = {
        "passed": False,
        "registration": digest(REG),
        "source_sha256": digest(SOURCE),
        "weights": s["weights"],
    }
    try:
        m, t, _ = load_trained(s["candidate"])
        end = core.atom(t, "<|im_end|>")
        for block in (s["baseline"], s["treatment"]):
            generate(
                m,
                t,
                t.decode(ids[:1024]),
                1,
                prefill_block=block,
                progress=lambda _: b.check(),
                stop_ids=[end],
            )
        for n in s["lengths"]:
            ref = None
            for repeat, mode in enumerate(
                ("baseline", "treatment", "treatment", "baseline")
            ):
                text = t.decode(ids[:n])
                if t.encode(text, add_special_tokens=False) != ids[:n]:
                    raise ValueError("Prefix tokenizer boundary differs")
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                capture = VocabularyCapture(m)
                block = s[mode]
                v = generate(
                    capture,
                    t,
                    text,
                    s["cap"],
                    prefill_block=block,
                    progress=lambda _: b.check(),
                    stop_ids=[end],
                )
                del v["prompt"]
                values = [
                    np.asarray(z.astype(mx.float32)).copy() for z in capture.outputs
                ]
                if ref is None:
                    ref = values
                errors = [
                    float(
                        np.sqrt(
                            np.sum((a - c) ** 2, dtype=np.float64)
                            / np.sum(a * a, dtype=np.float64)
                        )
                    )
                    for a, c in zip(ref, values)
                ]
                rows.append(
                    dict(
                        **v,
                        block=block,
                        repeat=repeat,
                        input_sha256=fingerprint(ids[:n]),
                        relative_rms=errors,
                    )
                )
                save(out / "runs.json", rows)
                del capture, values
        result.update(status="completed", lengths=summarize(rows, s))
        result["passed"] = all(v["passed"] for v in result["lengths"].values())
    except (InterruptedError, MemoryError, TimeoutError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
