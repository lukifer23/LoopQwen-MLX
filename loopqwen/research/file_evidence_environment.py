"""Record the code and native environment used for the changed-file training comparison."""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from ..io import digest, read, save

SOURCES = (
    "loopqwen/research/file_evidence.py",
    "loopqwen/research/file_evidence_fit.py",
    "loopqwen/research/native_function_study.py",
    "loopqwen/research/reading_pairs.py",
    "loopqwen/file_evidence.py",
    "loopqwen/looped_model.py",
    "loopqwen/tool_session.py",
    "loopqwen/tool_model.py",
    "loopqwen/tool_compare.py",
    "pyproject.toml",
    "uv.lock",
)


def capture(root, paths=SOURCES):
    path = root / "environment.json"
    sources = {k: digest(Path(k)) for k in paths}
    registration = digest(root / "registration.json")
    if path.exists():
        prior = read(path)
        if (
            sources != prior["execution_sources"]
            or registration != prior["registration_sha256"]
        ):
            raise ValueError(
                "The interrupted run used different native training or comparison code"
            )
        return

    def command(*args):
        return subprocess.check_output(args, text=True).strip()

    revision = command("git", "rev-parse", "HEAD")
    overlays = {}
    for k, sha in sources.items():
        blob = subprocess.run(["git", "show", revision + ":" + k], capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != sha:
            overlays[k] = Path(k).read_text()
    save(
        path,
        {
            "execution_revision": revision,
            "registration_sha256": registration,
            "architecture": platform.machine(),
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "physical_memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
            "macos_build": command("sw_vers", "-buildVersion"),
            "macos": platform.mac_ver()[0],
            "python": sys.version.split()[0],
            "libraries": {
                k: importlib.metadata.version(k)
                for k in (
                    "mlx",
                    "mlx-lm",
                    "transformers",
                    "safetensors",
                    "numpy",
                    "huggingface-hub",
                )
            },
            "execution_sources": sources,
            **({"working_source_text": overlays} if overlays else {}),
        },
    )
