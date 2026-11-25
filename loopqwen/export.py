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
from .data.build_data import digest
from .deployment import load_trained
from .io import checkpoint_location, read, save
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
            export_schema=2,
            supported_depths=[1, 2, 3],
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
            adapter_sha256=digest(
                checkpoint_location(checkpoint) / "adapters.safetensors"
            ),
            export_git=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            step=state["step"],
            validation_prompt=prompt,
        )
        if "config" in state:
            provenance.update(
                train_git=state["git"],
                config=state["config"],
                dataset=state["manifest"],
                versions=state["hardware"]["versions"],
            )
        else:
            spec = state["spec"]
            provenance.update(
                config=spec["deployment"]["config"],
                training_spec={k: v for k, v in spec.items() if k != "deployment"},
                training_identity=state["identity"],
                parent_weights=state["parent"],
                checkpoint_metadata_sha256=digest(
                    checkpoint_location(checkpoint) / "state.json"
                ),
            )
        has_reader = getattr(model, "attention_reader", None) is not None
        if has_reader:
            generation = checkpoint_location(checkpoint)
            for filename in ("attention-reader.json", "attention-reader.safetensors"):
                shutil.copyfile(generation / filename, temp / filename)
            config["attention_reader_metadata_sha256"] = digest(
                temp / "attention-reader.json"
            )
            provenance["attention_reader"] = read(temp / "attention-reader.json")
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
            "offloaded_prefill.py",
        ):
            shutil.copyfile(Path(__file__).with_name(filename), runtime / filename)
        if has_reader:
            shutil.copyfile(
                Path(__file__).with_name("attention_memory.py"),
                runtime / "attention_memory.py",
            )
        evaluation = runtime / "evaluation"
        evaluation.mkdir()
        (evaluation / "__init__.py").write_text("")
        shutil.copyfile(
            Path(__file__).parent / "evaluation" / "generation_eval.py",
            evaluation / "generation_eval.py",
        )
        (runtime / "__init__.py").write_text(
            "from .export_runtime import load_export\n"
            "from .evaluation.generation_eval import generate\n"
        )
        (temp / "README.md").write_text(
            "# Looped Qwen text export\n\n"
            "Run in this directory with the pinned native MLX environment. "
            "The generator initializes a fresh request cache, applies the checkpoint's "
            "attention reader when present, and preserves the full input. "
            "Pass an existing scratch directory as `offload_dir` for disk-backed BF16 caches.\n\n"
            "```python\n"
            "from runtime import generate, load_export\n\n"
            "model, tokenizer = load_export('.')\n"
            "prompt = tokenizer.apply_chat_template(\n"
            "    [{'role': 'user', 'content': 'What is 17 times 23?'}],\n"
            "    tokenize=False, add_generation_prompt=True, enable_thinking=True,\n"
            ")\n"
            "end = tokenizer.encode('<|im_end|>', add_special_tokens=False)\n"
            "result = generate(model, tokenizer, prompt, 256, stop_ids=end)\n"
            "print(result['text'])\n"
            "```\n"
        )
        decoder_parameters = count(model)
        reader_parameters = count(model.attention_reader.scorer) if has_reader else 0
        audit = dict(
            prompt=prompt,
            input_ids=ids.tolist(),
            last_top1=mx.argmax(after[0, -1]).item(),
            logits_sha256=hashlib.sha256(np.asarray(after).tobytes()).hexdigest(),
            merge_relative_rms=rms,
            unique_parameters=decoder_parameters + reader_parameters,
            fused_modules=fused,
            step=state["step"],
        )
        if has_reader:
            audit.update(
                decoder_parameters=decoder_parameters,
                reader_parameters=reader_parameters,
            )
            from .evaluation.generation_eval import generate

            body = "".join(
                f"Register entry {i} has code {100000 + 137 * i}.\n" for i in range(128)
            )
            message = body + "Which code belongs to register entry 37?"
            probe = tok.apply_chat_template(
                [{"role": "user", "content": message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            sample = generate(model, tok, probe, 4)
            audit["reader_probe"] = {
                "prompt": probe,
                "section": sample["attention_memory"],
                "token_ids": sample["token_ids"],
            }
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
