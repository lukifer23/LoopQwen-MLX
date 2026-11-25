"""Serialize a shared dense checkpoint and both exact native recurrent adapter banks."""

import hashlib
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from . import REVISION
from .io import checkpoint_location, digest, read, save
from .model_loader import local_snapshot
from .worker import child_lease


def write(temp, gen, model, t, state):
    temp = Path(temp)
    policy = model.context_policy
    source = local_snapshot()
    original = checkpoint_location(gen)
    policy.ordinary.text.save_weights(str(temp / "model.safetensors"))
    meta = read(original / "context-policy.json")
    for name in (*meta["files"], "context-policy.json"):
        shutil.copyfile(original / name, temp / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "LICENSE",
    ):
        shutil.copyfile(source / name, temp / name)
    dep = state["spec"]["deployment"]
    config = {
        "model_type": "qwen3_5_looped",
        "export_schema": 4,
        "base_revision": REVISION,
        "span": asdict(model.span),
        "text_config": read(source / "config.json")["text_config"],
        "weight_sharing": True,
        "recurrent_cache_version": 1,
        "alpha_override": model.alpha_override,
        "adapter_config": dep["config"],
        "context_policy_sha256": digest(temp / "context-policy.json"),
    }
    save(temp / "config.json", config)
    rt = temp / "runtime"
    rt.mkdir()
    for name in (
        "worker.py",
        "export_runtime.py",
        "looped_model.py",
        "recurrence.py",
        "recurrent_cache.py",
        "offloaded_prefill.py",
        "attention_memory.py",
        "deliberation.py",
        "context_policy.py",
        "io.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), rt / name)
    for name in ("evaluation/generation_eval.py", "training/adapters.py"):
        out = rt / name
        out.parent.mkdir(exist_ok=True)
        (out.parent / "__init__.py").touch()
        shutil.copyfile(Path(__file__).parent / name, out)
    (rt / "__init__.py").write_text(
        f"REVISION = {REVISION!r}\nfrom .export_runtime import load_export\nfrom .evaluation.generation_eval import generate\n"
    )
    prompt = "The report lists eight samples. How many samples are there?"
    ids = t.encode(prompt)
    logits = {}
    for name, net in [
        ("register", model),
        ("ordinary", policy.ordinary),
        ("prose", policy.prose),
    ]:
        y = net(mx.array([ids])).astype(mx.float32)
        logits[name] = hashlib.sha256(np.asarray(y).tobytes()).hexdigest()
    all_arrays = {}
    for net in (
        model,
        policy.ordinary,
        policy.prose,
        policy.classifier,
        model.attention_reader.scorer,
        policy.prose.attention_reader.scorer,
    ):
        for _, v in tree_flatten(net.parameters()):
            all_arrays[id(v)] = v
    count = sum(v.size for v in all_arrays.values())
    audit = {
        "prompt": prompt,
        "input_ids": [ids],
        "logits_sha256": logits["register"],
        "bank_logits": logits,
        "unique_parameters": count,
    }
    save(temp / "verification.json", audit)
    (temp / "README.md").write_text(
        '# Context-conditioned recurrent Qwen\n\nLoad this native MLX bundle with the included runtime. The two recurrent adapters share one dense Base checkpoint. The generator selects its decoder from the actual input and the requested output budget.\n\n```python\nfrom runtime import load_export, generate\nmodel, tokenizer = load_export(".")\n```\n'
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys;sys.path.insert(0,'.');from runtime.export_runtime import verify;verify('.')",
        ],
        cwd=temp,
        check=True,
        **child_lease(),
    )
    audit["fresh_process_verified"] = True
    save(temp / "verification.json", audit)
    return {
        "modules": 0,
        "parameters": count,
        "format": "shared-base-and-native-adapters",
    }
