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
    policy = getattr(model, "context_policy", None)
    ordinary = policy.ordinary if policy else model.ordinary_depth
    source = local_snapshot()
    original = checkpoint_location(gen)
    ordinary.text.save_weights(str(temp / "model.safetensors"))
    if policy:
        meta = read(original / "context-policy.json")
        files = [*meta["files"], "context-policy.json"]
        mode = {
            "export_schema": 4,
            "context_policy_sha256": digest(original / "context-policy.json"),
        }
    else:
        files = [
            "adapters.safetensors",
            "deliberation.json",
            "exit-readout.safetensors",
        ]
        mode = {
            "export_schema": 5,
            "deliberation_sha256": digest(original / "deliberation.json"),
        }
    for name in files:
        (temp / name).parent.mkdir(parents=True, exist_ok=True)
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
        **mode,
        "base_revision": REVISION,
        "span": asdict(model.span),
        "text_config": read(source / "config.json")["text_config"],
        "weight_sharing": True,
        "recurrent_cache_version": 1,
        "alpha_override": model.alpha_override,
        "adapter_config": dep["config"],
    }
    save(temp / "config.json", config)
    provenance = {
        "base_revision": REVISION,
        "base_shards": {
            path.name: digest(path) for path in source.glob("*.safetensors")
        },
        "checkpoint_state_sha256": digest(original / "state.json"),
        "checkpoint_step": state["step"],
        "checkpoint_identity": state["identity"],
        "files": {name: digest(temp / name) for name in files},
        "export_git": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fusion": "unmerged native adapter tensors",
    }
    save(temp / "provenance.json", provenance)
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
        "conditional.py",
        "evidence_loop.py",
        "native_evidence.py",
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
    views = {
        "recurrent": model,
        **(policy.views() if policy else {"ordinary": ordinary}),
    }
    extras = [policy.classifier] if policy else []
    readers = [
        (name, route)
        for name, route in (
            ("evidence", "deduction"),
            ("induction", "native_induction"),
        )
        if getattr(policy, name, None) is not None
    ]
    for name, _ in readers:
        reader = getattr(policy, name)
        extras.extend([reader.reader, reader.gate])
        if getattr(reader, "intent", None) is not None:
            extras.append(reader.intent)
    for bank in views.values():
        reader = getattr(bank, "attention_reader", None)
        if reader is not None:
            extras.append(reader.scorer)
    for name, net in views.items():
        y = net(mx.array([ids])).astype(mx.float32)
        logits[name] = hashlib.sha256(np.asarray(y).tobytes()).hexdigest()
    all_arrays = {}
    for bank in views.values():
        head = getattr(bank, "exit_readout", None)
        if head is not None:
            all_arrays[id(head)] = head
    for net in [*views.values(), *extras]:
        for _, v in tree_flatten(net.parameters()):
            all_arrays[id(v)] = v
    count = sum(v.size for v in all_arrays.values())
    audit = {
        "prompt": prompt,
        "input_ids": [ids],
        "logits_sha256": logits["recurrent"],
        "unique_parameters": count,
        "adapter_banks": len(views) - 1,
    }
    if policy:
        audit["bank_logits"] = {
            "register": logits["recurrent"],
            **{n: logits[n] for n in policy.views()},
        }
    else:
        audit["ordinary_logits_sha256"] = logits["ordinary"]
        audit["readout_sha256"] = model.deliberation_policy["readout_sha256"]
    for name, route in readers:
        from .evaluation.generation_eval import generate

        prompt = read(original / name / "scope.json")["example"]
        sample = generate(model, t, prompt, 32, stop_ids=(248046,))
        if sample["context_route"] != route:
            raise ValueError("Export probe missed the learned evidence route")
        audit[name + "_probe"] = {"prompt": prompt, "tokens": sample["token_ids"]}
    save(temp / "verification.json", audit)
    (temp / "README.md").write_text(
        '# Context-conditioned recurrent Qwen\n\nLoad this native MLX bundle with the included runtime. The ordinary and recurrent execution paths share one dense Base checkpoint. The bundled generator applies the trained decoding policy to the actual input.\n\n```python\nfrom runtime import load_export, generate\nmodel, tokenizer = load_export(".")\n```\n'
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
        "adapter_banks": audit["adapter_banks"],
    }
