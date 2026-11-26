"""Build an independent offline assistant using the qualified long register prefill policy."""

import os
import shutil
import tempfile
from pathlib import Path

from loopqwen.io import digest, read, save

RUNTIME = ("page_prefill.py", "evaluation/generation_eval.py", "export_runtime.py")


def build(source, parent, destination, full):
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
        save(staging / "config.json", cfg)
        for name in RUNTIME:
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "loopqwen" / name,
                staging / "runtime" / name,
            )
        with (staging / "README.md").open("a") as f:
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
