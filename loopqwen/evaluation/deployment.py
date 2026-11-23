"""Compare the unmerged research reference with its standalone BF16 deployment."""

from dataclasses import replace
import gc
from pathlib import Path

import mlx.core as mx

from .. import checkpoint
from ..data.v2 import rows
from ..export import export
from ..export_runtime import load_export
from ..io import read, save, digest
from ..training.losses import validation_nll
from .phase2 import predictions
from .trace_verifier import verify


def audit(source, out, report):
    source = checkpoint.location(source)
    out = Path(out)
    if not out.exists():
        export(source, out, "fp32")
    if read(out / "provenance.json")["adapter_sha256"] != digest(
        source / "adapters.safetensors"
    ):
        raise ValueError("Merged weights belong to another adapter")
    data = rows("validation", "trace")[:24]
    prose = rows("language_validation", "language")[:24]
    results = {}
    for format in ("adapter", "merged"):
        m, t = (
            checkpoint.load_trained(source)[:2]
            if format == "adapter"
            else load_export(out)
        )
        result = {}
        for k in (1, 2):
            m.span = replace(m.span, passes=k)
            run = predictions(m, t, data, 192, padded=True, detailed=True)
            for rec, row in zip(run["records"], data, strict=True):
                cert = verify(row, rec["text"], rec["token_cap_reached"])
                rec["correct"] = cert["final"] == "correct"
                rec["verified"] = cert["verified"]
            result[str(k)] = dict(
                correct=sum(r["correct"] for r in run["records"]),
                records=run["records"],
                retention=validation_nll(m, prose),
                alpha=float(m.effective_alpha()) if k == 2 else None,
            )
        results[format] = result
        del m
        gc.collect()
        mx.clear_cache()
    diffs = {}
    for depth, a in results["adapter"].items():
        b = results["merged"][depth]
        if a["alpha"] != b["alpha"]:
            raise ValueError("Merged gate changed")
        diffs[depth] = sum(
            x["generated_ids"] != y["generated_ids"]
            for x, y in zip(a["records"], b["records"], strict=True)
        )
    save(
        report,
        dict(
            source=digest(source / "adapters.safetensors"),
            verification=read(out / "verification.json"),
            different_tokens_by_request=diffs,
            results=results,
        ),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "destination", "report"):
        p.add_argument(key)
    a = p.parse_args()
    audit(a.source, a.destination, a.report)
