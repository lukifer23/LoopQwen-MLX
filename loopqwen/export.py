import gc
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.tuner.lora import LoRALinear

from . import MODEL, REVISION
from .checkpoint import load_trained, location
from .data.build_data import digest
from .io import read, save
from .model_loader import count, local_snapshot


def export(checkpoint, output, fusion="stock"):
    if fusion not in ("stock", "fp32"):
        raise ValueError("Unknown fusion dtype")
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Choose a new export directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".export-", dir=out.parent))
    try:
        model, tok, state = load_trained(checkpoint)
        prompt = "Question: 17 × 23\nAnswer:"
        ids = mx.array([tok.encode(prompt)])
        before = model(ids).astype(mx.float32)
        mx.eval(before)
        fused = 0
        for layer in model.text.layers:
            for name, module in list(layer.named_modules()):
                if isinstance(module, LoRALinear):
                    parent, key = name.rsplit(".", 1)
                    if fusion == "fp32":
                        linear = module.linear
                        delta = (
                            module.scale * module.lora_b.astype(mx.float32).T
                        ) @ module.lora_a.astype(mx.float32).T
                        linear.weight = (
                            linear.weight.astype(mx.float32) + delta
                        ).astype(linear.weight.dtype)
                    else:
                        linear = module.fuse()
                    setattr(getattr(layer, parent), key, linear)
                    fused += 1
        after = model(ids).astype(mx.float32)
        rms = mx.sqrt(mx.mean((after - before) ** 2) / mx.mean(before**2)).item()
        if not rms < 0.03:
            raise ValueError(f"BF16 merge failed parity: relative RMS {rms}")
        model.save_weights(str(temp / "model.safetensors"))
        source = local_snapshot()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
            "LICENSE",
        ):
            shutil.copyfile(source / name, temp / name)
        config = dict(
            model_type="qwen3_5_looped",
            base_model=MODEL,
            base_revision=REVISION,
            span=asdict(model.span),
            text_config=read(source / "config.json")["text_config"],
            weight_sharing=True,
            recurrent_cache_version=1,
            alpha_override=model.alpha_override,
        )
        provenance = dict(
            fusion=fusion,
            base_revision=REVISION,
            base_shards={p.name: digest(p) for p in source.glob("*.safetensors")},
            adapter_sha256=digest(location(checkpoint) / "adapters.safetensors"),
            train_git=state["git"],
            export_git=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            config=state["config"],
            dataset=state["manifest"],
            versions=state["hardware"]["versions"],
            step=state["step"],
            validation_prompt=prompt,
        )
        save(temp / "provenance.json", provenance)
        save(temp / "config.json", config)
        runtime = temp / "runtime"
        runtime.mkdir()
        for filename in (
            "worker.py",
            "export_runtime.py",
            "looped_model.py",
            "recurrence.py",
            "recurrent_cache.py",
        ):
            shutil.copyfile(Path(__file__).with_name(filename), runtime / filename)
        (runtime / "__init__.py").write_text(
            "from .export_runtime import load_export\n"
        )
        (temp / "README.md").write_text(
            "# Looped Qwen text export\n\nIn this directory use the pinned project environment. "
            "Run `from runtime import load_export` and `model, tokenizer = load_export('.')`. "
            "Call `model.make_cache()` once per prompt.\n"
        )
        audit = dict(
            prompt=prompt,
            input_ids=ids.tolist(),
            last_top1=mx.argmax(after[0, -1]).item(),
            logits_sha256=hashlib.sha256(np.asarray(after).tobytes()).hexdigest(),
            merge_relative_rms=rms,
            unique_parameters=count(model),
            fused_modules=fused,
            step=state["step"],
        )
        save(temp / "verification.json", audit)
        del model, before, after
        gc.collect()
        mx.clear_cache()
        from .worker import child_lease

        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; sys.path.insert(0, '.'); from runtime.export_runtime import verify; verify('.')",
            ],
            **child_lease(),
            cwd=temp,
            check=True,
        )
        audit["fresh_process_verified"] = True
        save(temp / "verification.json", audit)
        temp.rename(out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return dict(
        directory=str(out),
        modules=fused,
        relative_rms=rms,
        parameters=audit["unique_parameters"],
    )
