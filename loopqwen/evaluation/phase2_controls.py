"""Unadapted development baselines, representation diagnostics and external transfer."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from ..checkpoint import load_trained
from ..data.build_data import GSM_REV
from ..data.math500 import numeric_slice
from ..data.tokenize import encode
from ..data.v2 import rows
from ..io import save
from ..looped_model import LoopedModel
from ..model_loader import load_base
from .phase2 import manifest, nll, predictions, score


def representation(model, x):
    h = model.text.model.embed_tokens(x)
    seen = []
    for i, p in model.sites:
        layer = model.text.layers[i]
        if i == model.span.start:
            prev = h
            if p == 0:
                seen.append(h)
        h = layer(
            h,
            mask=create_ssm_mask(h, None)
            if layer.is_linear
            else create_attention_mask(h, None),
        )
        if i == model.span.start + model.span.length - 1:
            if model.span.mode == "stabilized" and p:
                alpha = (
                    model.alpha_override
                    if model.alpha_override is not None
                    else mx.sigmoid(model.alpha_logit)
                )
                h = (prev + alpha * (h - prev)).astype(h.dtype)
            seen.append(h)
    prev = seen[0].astype(mx.float32)
    values = []
    for v in seen[1:]:
        v = v.astype(mx.float32)
        values.append(
            dict(
                rms=mx.sqrt(mx.mean(v * v)).item(),
                max_abs=mx.max(mx.abs(v)).item(),
                relative_change=(
                    mx.linalg.norm(v - prev) / mx.linalg.norm(prev)
                ).item(),
                cosine=(
                    mx.sum(v * prev) / (mx.linalg.norm(v) * mx.linalg.norm(prev))
                ).item(),
            )
        )
        prev = v
    return values


def development():
    base, tok, info = load_base()
    m = LoopedModel(base)
    m.eval()
    settings = manifest("phase2_direct")
    dev = rows("validation", "answer")
    result = {}
    for k, start in ((1, 10), (2, 8), (2, 10), (2, 16)):
        m.span = replace(m.span, passes=k, start=start)
        result[f"{start}/{k}"] = dict(
            metrics=score(m, tok, settings, "validation"),
            representations=[
                representation(m, mx.array([r["ids"][: r["prefix"]]])) for r in dev[:5]
            ],
        )
    save("outputs/phase2/base_dev.json", result)


def transfer():
    design = manifest("phase2_confirm")
    # External tests are exploratory transfer checks evaluated only after design freeze.
    ext = Path("data/phase2_external")
    ext.mkdir(exist_ok=True)
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/test.jsonl"
    source = ext / "gsm_test.jsonl"
    if not source.exists():
        source.write_bytes(urlopen(url, timeout=30).read())
    result = {
        "gsm_revision": GSM_REV,
        "gsm_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runs": {},
    }
    for checkpoint in design["matched"][0]:
        m, tok, _ = load_trained(Path("checkpoints") / checkpoint)
        gsm = []
        for i, line in enumerate(source.read_text().splitlines()[:90]):
            r = json.loads(line)
            prompt = r["question"] + "\nAnswer:"
            gsm.append(
                encode(
                    dict(
                        id=f"gsm-test-{i}",
                        category="gsm",
                        steps=0,
                        prompt=prompt,
                        answer=r["answer"]
                        .rsplit("####", 1)[1]
                        .strip()
                        .replace(",", ""),
                    ),
                    tok,
                )
            )
        public = {"gsm": gsm, "math500": numeric_slice(tok, 60, ext / "math500")}
        result["runs"][checkpoint] = {}
        for name, dataset in public.items():
            for r in dataset:
                r.update(complexity="unrated", input_length=len(r["ids"]) - 1)
            result["runs"][checkpoint][name] = dict(
                loss=nll(m, dataset), generations=predictions(m, tok, dataset, 16)
            )
        del m
        mx.clear_cache()
    base, tok, _ = load_base()
    model = LoopedModel(base)
    model.eval()
    model.span = replace(model.span, passes=1)
    result["base_prose_test"] = nll(
        model, rows("language_test", "language")[: design["language_count"]]
    )
    save("outputs/phase2/transfer.json", result)


if __name__ == "__main__":
    import sys

    {"dev": development, "transfer": transfer}[sys.argv[1]]()
