"""Export a true K=1 adapter loadable by the stock mlx_lm.load API."""

import gc
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from .data.v2 import rows
from .deployment import load_trained
from .evaluation.trace_verifier import verify
from .io import checkpoint_location, digest, read, save
from .model_loader import local_snapshot
from .worker import child_lease

# Run in isolated mode: this script uses only MLX, MLX-LM and standard Python.
STOCK = """
import json,sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
root=Path(sys.argv[2])
model,tok=load(sys.argv[1],adapter_path=str(root))
result=[]
for x in json.loads((root/"requests.json").read_text()):
    ids=tok.encode(x["prompt"],add_special_tokens=False)
    assert ids==x["ids"]
    cache=model.make_cache()
    q=mx.array([ids])
    generated=[]
    for i in range(x["cap"]):
        logits=model(q,cache=cache)[:,-1]
        if not i:
            ref=mx.load(str(root/x["reference"]))
            d=logits.astype(mx.float32)-ref
            rms=mx.sqrt(mx.mean(d*d)/mx.mean(ref*ref)).item()
        token=mx.argmax(logits,axis=-1).item()
        if token in tok.eos_token_ids:break
        generated.append(token)
        q=mx.array([[token]])
    result.append({"rms":rms,"tokens":generated,"text":tok.decode(generated)})
print(json.dumps(result))
"""


def export_stock(source, destination):
    gen = checkpoint_location(source)
    if (gen / "attention-reader.json").exists():
        raise ValueError(
            "Stock K1 cannot represent the learned attention mask. Use the recurrent export."
        )
    out = Path(destination)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True, parents=True)
    state = read(gen / "state.json")
    c = (state if "config" in state else state["spec"]["deployment"])["config"]
    model, tok, _ = load_trained(gen)
    model.span = replace(model.span, passes=1)
    samples = rows("validation", "trace")[:16]
    weights = mx.load(str(gen / "adapters.safetensors"))
    names = {}
    for key, value in weights.items():
        if key == "alpha_logit":
            continue
        if not key.startswith("text.model.layers.") or key.rsplit(".", 1)[-1] not in (
            "lora_a",
            "lora_b",
        ):
            raise ValueError("Unsupported K=1 trainable tensor " + key)
        names["language_model." + key.removeprefix("text.")] = value
    with TemporaryDirectory(dir=out.parent) as temp:
        work = Path(temp) / "adapter"
        work.mkdir()
        mx.save_safetensors(str(work / "adapters.safetensors"), names)
        save(
            work / "adapter_config.json",
            dict(
                fine_tune_type="lora",
                num_layers=0,
                lora_parameters=dict(
                    rank=c["rank"],
                    scale=c["lora_alpha"] / c["rank"],
                    dropout=0,
                    keys=sorted({key.rsplit(".", 1)[0] for key in names}),
                ),
            ),
        )
        requests = []
        for i, row in enumerate(samples):
            ids = row["ids"][: row["prefix"]]
            cap = 192
            q = mx.array([ids])
            generated = []
            cache = model.make_cache()
            name = f"logits-{i}.npy"
            for j in range(cap):
                logits = model.project(model.hidden(q, cache)[:, -1])
                if j == 0:
                    mx.save(str(work / name), logits.astype(mx.float32))
                token = mx.argmax(logits, axis=-1).item()
                if token in tok.eos_token_ids:
                    break
                generated.append(token)
                q = mx.array([[token]])
            requests.append(
                dict(
                    id=row["id"],
                    prompt=row["prompt"],
                    ids=ids,
                    reference=name,
                    cap=cap,
                    tokens=generated,
                    text=tok.decode(generated),
                )
            )
        del model, weights, names
        gc.collect()
        mx.clear_cache()
        save(work / "requests.json", requests)
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                STOCK,
                str(local_snapshot()),
                str(work.resolve()),
            ],
            check=True,
            text=True,
            capture_output=True,
            **child_lease(),
        )
        actual = json.loads(proc.stdout)
        stats = []
        for row, req, res in zip(samples, requests, actual, strict=True):
            stats.append(
                dict(
                    id=row["id"],
                    max_context_logits_rms=res["rms"],
                    equal_tokens=req["tokens"] == res["tokens"],
                    reference=verify(row, req["text"])["final"],
                    stock=verify(row, res["text"])["final"],
                    stock_text=res["text"],
                )
            )
        if any(
            r["max_context_logits_rms"] > 0.001 or not r["equal_tokens"] for r in stats
        ):
            raise ValueError(f"Stock K=1 parity failed: {stats}")
        for req in requests:
            (work / req["reference"]).unlink()
        (work / "requests.json").unlink()
        save(
            work / "deployment.json",
            dict(
                schema=1,
                base=c["base_model"],
                revision=c["base_revision"],
                source=digest(gen / "adapters.safetensors"),
                weights=digest(work / "adapters.safetensors"),
                stage=state["step"],
                deployment_depth=1,
                validation=stats,
            ),
        )
        work.rename(out)
    return str(out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("destination")
    args = p.parse_args()
    print(export_stock(args.source, args.destination))
