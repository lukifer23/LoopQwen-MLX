"""Unadapted development baselines, representation diagnostics and external transfer."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx

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
    previous = None
    values = []

    def observe(event, recurrent_pass, activation):
        nonlocal previous
        v = mx.stop_gradient(activation).astype(mx.float32)
        if event == "entry":
            previous = v
            return
        prev = previous
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
        previous = None

    mx.eval(model.hidden(x, observer=observe))
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
