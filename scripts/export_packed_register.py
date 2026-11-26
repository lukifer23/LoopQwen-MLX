"""Build an independent offline assistant using the qualified long register prefill policy."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loopqwen.io import digest, read, save

RUNTIME = ("page_prefill.py", "evaluation/generation_eval.py", "export_runtime.py")
RUNTIME_REVISION = "0880d1e830e8013aa3a1591eb3967273928796c2"


COMPACT_REVISION = "9a39af860322ee28b446315980a5f1b8845a7da4"
FUSED_REVISION = "6faf8a27e107f1021dcb13c3eacbe48560f0cc4c"
COMPACT_RUNTIME = (
    "export_runtime.py",
    "register_session.py",
    "mixed_register.py",
    "page_prefill.py",
    "evaluation/generation_eval.py",
    "kernels/packed_decode.py",
)


def build(source, parent, destination, full, *, compact=False):
    original = read(parent / "export-files.json")
    if (
        not read(parent / "decision.json")["passed"]
        or not read(full / "decision.json")["passed"]
    ):
        raise ValueError("The release and physical 131K comparison must both qualify")
    if read(source / "tool-export-files.json") != original:
        raise ValueError(
            "Source export inventory differs from its archived qualification"
        )
    if any(digest(source / k) != h for k, h in original.items()):
        raise ValueError("A source release file differs from its qualified digest")
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        # Copy precisely the certified files, ignoring generated Python caches.
        for name in original:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        cfg = read(staging / "config.json")
        cfg["register_prefill"] = "direct-packed-register-v1"
        if compact:
            cfg["register_weights"] = "affine8-mlp-output-exact-input-v1"
        save(staging / "config.json", cfg)
        repository = Path(__file__).resolve().parents[1]
        for name in COMPACT_RUNTIME if compact else RUNTIME:
            revision = RUNTIME_REVISION
            if compact:
                revision = (
                    FUSED_REVISION
                    if name == "kernels/packed_decode.py"
                    else COMPACT_REVISION
                )
            payload = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository),
                    "show",
                    f"{revision}:loopqwen/{name}",
                ]
            )
            target = staging / "runtime" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        with (staging / "README.md").open("a") as f:
            if compact:
                f.write(
                    "\n## Long register weights\n\nThe trained router selects the recurrent register decoder. "
                    "At 32,768 input tokens or more this decoder converts the feed-forward and output weights to eight bits, "
                    "retains exact BF16 input rows and uses fused full-history decoding. Subsequent requests reload the "
                    "original language weights when needed. Every vocabulary ID remains available during generation.\n"
                )
            else:
                f.write(
                    "\n## Long register prefill\n\nFor resident register requests of at least 16,384 tokens, "
                    "the learned context selector enables native 8-bit paged prefill. Each of the eight "
                    "recurrent global sites retains the complete input and output history. "
                    "The final 512 input tokens and all output tokens use full-history attention.\n"
                )
        files = {
            str(p.relative_to(staging)): digest(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        save(staging / "tool-export-files.json", files)
        if destination.exists():
            if read(destination / "tool-export-files.json") != files or any(
                digest(destination / k) != h for k, h in files.items()
            ):
                raise FileExistsError("Existing bundle differs from this exact release")
        else:
            os.rename(staging, destination)
    return files
