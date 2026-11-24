"""Publish an immutable snapshot of measured Phase 7 artifacts."""

import argparse
import shutil
import tempfile
from pathlib import Path
from loopqwen.training.exposure import ARM, OUT, REG, S, parent_lineage
from loopqwen.training.mechanism import generation
from loopqwen.io import read, save, digest
from verify_phase7_archive import verify


def archive(dest):
    if dest.exists():
        raise FileExistsError(dest)
    d = read(OUT / "decision.json")
    gen = generation(ARM, d["step"])
    lineage = parent_lineage()
    state = read(gen / "state.json")
    if state["lineage"] != lineage:
        raise ValueError("Parent changed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        root = Path(tmp) / "release"
        shutil.copytree(OUT, root)
        shutil.copy(REG, root / "registration.json")
        model = root / "model"
        model.mkdir()
        for f in ("adapters.safetensors", "state.json"):
            shutil.copy(gen / f, model / f)
        save(
            root / "recovery.json",
            dict(
                generation=str(gen),
                optimizer_sha256=digest(gen / "optimizer.safetensors"),
                parent=lineage,
            ),
        )
        save(
            root / "checksums.json",
            {
                str(p.relative_to(root)): digest(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        )
        verify(root)
        root.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=Path("docs/measurements/phase7"))
    archive(p.parse_args().destination)
