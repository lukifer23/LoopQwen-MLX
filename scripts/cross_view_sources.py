"""Fetch the pinned reading corpora and optional training-only teacher as atomic files."""

import argparse
import tempfile
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

from loopqwen.io import digest, read
from scripts.cross_view_material import SPEC


def fetch(target, url, sha):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=target.parent) as d:
            stage = Path(d) / target.name
            with urlopen(url, timeout=120) as inp, stage.open("wb") as out:
                copyfileobj(inp, out, 2**20)
            if digest(stage) != sha:
                raise ValueError("Different downloaded bytes: " + target.name)
            stage.replace(target)
    if digest(target) != sha:
        raise ValueError("Different local bytes: " + target.name)
    print(target.name, sha, flush=True)


def install(data_only):
    s = read(SPEC)
    root = Path("data/research_external")
    fetch(root / "squad-train-v1.1.json", s["sources"][2], s["source_hashes"]["squad"])
    fetch(root / "quoref/train-dev.zip", s["sources"][1], s["source_hashes"]["quoref"])
    if not data_only:
        t = s["teacher"]
        base = "https://huggingface.co/" + t["repo"] + "/resolve/" + t["revision"] + "/"
        for name, sha in {
            **t["files"],
            "model.safetensors": t["weights_sha256"],
        }.items():
            fetch(Path(t["directory"]) / name, base + name, sha)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-only", action="store_true")
    install(p.parse_args().data_only)
