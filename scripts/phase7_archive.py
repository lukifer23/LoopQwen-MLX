"""Publish an immutable snapshot of measured Phase 7 artifacts."""

import argparse
import shutil
import tempfile
from pathlib import Path
from loopqwen.training.exposure import ARM, OUT, REG, parent_lineage
from loopqwen.training.mechanism import generation
from loopqwen.training.transfer7 import ARM_T
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
        transfer = read(OUT / "transfer/decision.json")
        shutil.copy(
            "experiments/phase7_transfer_parent.json",
            root / "transfer/registration.json",
        )
        if transfer["step"]:
            child = generation(ARM_T, transfer["step"])
            target = root / "transfer/model"
            target.mkdir()
            for filename in ("adapters.safetensors", "state.json"):
                shutil.copy(child / filename, target / filename)
            save(
                root / "transfer/recovery.json",
                dict(
                    generation=str(child),
                    optimizer_sha256=digest(child / "optimizer.safetensors"),
                ),
            )
        for directory, label, record, command, mode in [
            (model, "R624 categorical mechanism", state, "exposure", ""),
            *(
                [
                    (
                        target,
                        "Unrestricted next-digit transfer",
                        read(child / "state.json"),
                        "transfer",
                        "--vocabulary ",
                    )
                ]
                if transfer["step"]
                else []
            ),
        ]:
            config = record["config"]
            sha = digest(directory / "adapters.safetensors")
            data = record["provenance"]["base"]["data"]
            (directory / "README.md").write_text(
                f"# {label}\n\n"
                f"Experimental shared-depth adapter for {config['base_model']} at `{config['base_revision']}` (Apache 2.0). "
                "Use native MLX 0.32.4, MLX-LM 0.32.2 and the unmerged checkpoint. "
                "The 1,344,512 trained weights comprise 1,336,320 block LoRA weights and 8,192 shared readout weights. "
                "The original tied vocabulary head stays frozen. Use K=3 for three-hop tasks. "
                "Longer categorical paths degrade, and general-purpose chat improvement remains unestablished.\n\n"
                f"Trainable weights SHA-256 `{sha}`. Data identity `{data}`.\n\n"
                f"Local training stage: `scripts/run_phase7.sh {command}`. "
                "The exact training parent and AdamW policy are recorded in the stage registration and recovery files.\n\n"
                f"```sh\n.venv/bin/python scripts/phase7_predict.py {mode}--map 5 4 7 6 0 2 1 3 --start 0 --depth 3\n```\n\n"
                "See [Phase 7 results]("
                + (
                    "../../../../PHASE7_RESULTS.md"
                    if mode
                    else "../../../PHASE7_RESULTS.md"
                )
                + ") for actual training exposure, retained accuracy, inference cost and limitations.\n"
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
