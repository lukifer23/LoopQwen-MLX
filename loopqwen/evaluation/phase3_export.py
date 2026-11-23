"""Task-level audit of the selected 100K adapter and its standalone BF16 fusion."""

import gc
from pathlib import Path

import mlx.core as mx

from .. import checkpoint
from ..data.v2 import rows
from ..export import export
from ..export_runtime import load_export, verify
from ..io import read, save, digest
from ..training.losses import validation_nll
from .phase2 import predictions
from .generation_eval import generate


def run():
    name = "p3_" + read("outputs/phase3/selection.json")["selected"]
    root = Path("checkpoints") / name
    generation = root / read(root / "targets-100000.json")["directory"]
    out = Path("exports") / (name + "_100k_fp32_fusion")
    if not out.exists():
        export(generation, out, "fp32")
    if read(out / "provenance.json")["adapter_sha256"] != digest(
        generation / "adapters.safetensors"
    ):
        raise ValueError("Export belongs to a different generation")
    verify(out)
    samples = rows("validation", "trace")[:30]
    prose = rows("language_validation", "language")[:24]
    from dataclasses import replace

    result = {}
    for mode in ("adapter", "merged"):
        model, tok = (
            checkpoint.load_trained(generation)[:2]
            if mode == "adapter"
            else load_export(out)
        )
        if mode == "adapter":
            prompt = "Evaluate ((8 + 4) * 3 - 5).\nShow your steps. End with Final answer: <number>.\nSolution:"
            save("outputs/phase3/example.json", generate(model, tok, prompt, 96))
        gate = float(model.effective_alpha())
        result[mode] = {}
        for k in (1, 2):
            model.span = replace(model.span, passes=k)
            rr = predictions(model, tok, samples, 192, padded=True, detailed=True)
            result[mode][str(k)] = dict(
                gate=gate,
                trace_nll=validation_nll(model, samples),
                retention=validation_nll(model, prose),
                generations=rr,
            )
        del model
        gc.collect()
        mx.clear_cache()
    for k in ("1", "2"):
        a, b = [result[m][k]["generations"]["records"] for m in ("adapter", "merged")]
        result["different_strings_k" + k] = sum(
            x["generated_ids"] != y["generated_ids"] for x, y in zip(a, b)
        )
    save(
        "outputs/phase3/export-audit.json",
        dict(
            results=result,
            verification=read(out / "verification.json"),
            provenance=read(out / "provenance.json"),
        ),
    )


if __name__ == "__main__":
    run()
