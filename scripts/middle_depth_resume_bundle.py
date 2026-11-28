"""Retain both full-document optimizer boundaries and the exact four fitted feature shards."""

import argparse
import json
import shutil
from hashlib import sha256
from operator import xor
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.io import digest, read, save
from scripts.middle_depth_portable import add

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/measurements/middle-depth"
KIT = REPO / "research/results/middle-depth-full-document"
TRAIN = Path("checkpoints/middle-depth/long-optimizer/loop")
MEMBERS = (
    "state.json",
    "registration.json",
    "optimizer.safetensors",
    "adapters.safetensors",
)


def pack():
    source = REPO / TRAIN
    pointer = read(source / "latest.json")
    current, previous = pointer["path"], pointer["previous"]
    end = read(source / current / "state.json")
    prior = read(source / previous / "state.json")
    full_fit = read(EVIDENCE / "long-fit-loop.json")
    if (
        end["step"] != prior["step"] + 1
        or end["identity"] != prior["identity"]
        or end["log"][:-1] != prior["log"]
        or end["log"] != full_fit["log"]
        or end["hashes"]["adapters.safetensors"] != full_fit["weights"]
        or digest(source / current / "state.json") != pointer["state"]
    ):
        raise ValueError("Invalid optimizer history")
    archive = KIT / "resume.zip"
    index = {}
    with TemporaryDirectory(dir=KIT) as temp:
        stage = Path(temp) / archive.name
        with ZipFile(stage, "w") as out:
            weights = (source / current / "adapters.safetensors").read_bytes()

            def member(path, mode="literal"):
                data = (REPO / path).read_bytes()
                name = str(path)
                index[name] = {
                    "sha": sha256(data).hexdigest(),
                    "bytes": len(data),
                    "mode": mode,
                }
                if mode == "difference":
                    if len(data) != len(weights):
                        raise ValueError("Incompatible adapter file layouts")
                    add(out, name, bytes(map(xor, data, weights)))
                elif mode == "literal":
                    add(out, name, data)

            member(TRAIN / "latest.json")
            for step, state in ((previous, prior), (current, end)):
                for filename in MEMBERS:
                    path = TRAIN / step / filename
                    if (
                        filename != "state.json"
                        and digest(REPO / path) != state["hashes"][filename]
                    ):
                        raise ValueError(
                            f"Persisted optimizer boundary changed at {path}"
                        )
                    mode = "literal"
                    if filename == "adapters.safetensors":
                        mode = "deployed" if step == current else "difference"
                    member(path, mode)
            features = read(EVIDENCE / "long-features-loop.json")
            if features["parent"] != full_fit["parent"] or set(features["rows"]) != set(
                end["identity"]["order"]
            ):
                raise ValueError("Features differ from trained source order")
            for row in features["rows"].values():
                path = Path(row["file"])
                if (
                    digest(REPO / path) != row["metadata"]["sha"]
                    or read((REPO / path).with_suffix(".json")) != row["metadata"]
                ):
                    raise ValueError(f"Changed full-context feature {path}")
                member(path)
                member(path.with_suffix(".json"))
            add(out, "index.json", json.dumps(index, sort_keys=True).encode())
        stage.replace(archive)
    save(
        KIT / "resume.json",
        {
            "sha": digest(archive),
            "bytes": archive.stat().st_size,
            "restored_member_bytes": sum(v["bytes"] for v in index.values()),
            "weights": full_fit["weights"],
            "latest": current,
            "previous": previous,
            "steps": [prior["step"], end["step"]],
            "parent": full_fit["parent"],
            "features": digest(EVIDENCE / "long-features-loop.json"),
        },
    )


def restore(destination, previous_only=False):
    if destination.exists():
        raise FileExistsError(destination)
    archive = KIT / "resume.zip"
    receipt = read(KIT / "resume.json")
    if (
        digest(archive) != receipt["sha"]
        or digest(KIT / "changes.zip") != read(KIT / "changes.json")["sha"]
    ):
        raise ValueError("Invalid portable research archive")
    with ZipFile(KIT / "changes.zip") as deployment:
        weights = deployment.read("adapters.safetensors")
    if sha256(weights).hexdigest() != receipt["weights"]:
        raise ValueError("Changed deployed model")
    destination.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=destination.parent) as temp, ZipFile(archive) as z:
        root = Path(temp) / "workspace"
        for name, entry in json.loads(z.read("index.json")).items():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Invalid archive path")
            if previous_only and rel.is_relative_to(TRAIN / receipt["latest"]):
                continue
            mode = entry["mode"]
            if mode == "deployed":
                data = weights
            elif mode == "difference":
                patch = z.read(name)
                if len(patch) != len(weights):
                    raise ValueError("Invalid delta size")
                data = bytes(map(xor, patch, weights))
            elif mode == "literal":
                data = z.read(name)
            else:
                raise ValueError("Unsupported archive member")
            if len(data) != entry["bytes"] or sha256(data).hexdigest() != entry["sha"]:
                raise ValueError(name)
            path = root / rel
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
        if previous_only:
            step = receipt["previous"]
            save(
                root / TRAIN / "latest.json",
                {
                    "path": step,
                    "previous": None,
                    "state": digest(root / TRAIN / step / "state.json"),
                },
            )
        root.rename(destination)


def replay():
    """Re-execute the original trainer from the penultimate optimizer generation."""
    import os
    import subprocess
    import sys

    from safetensors.numpy import load

    with TemporaryDirectory() as temp:
        root = Path(temp) / "restored"
        restore(root, previous_only=True)
        for directory in ("research", "exports", "data", "experiments"):
            (root / directory).symlink_to(REPO / directory, target_is_directory=True)
        records = root / EVIDENCE.relative_to(REPO)
        records.mkdir(parents=True)
        (records / "charges").symlink_to(EVIDENCE / "charges", target_is_directory=True)
        shutil.copyfile(EVIDENCE / "sparse-parity.json", records / "sparse-parity.json")
        policy = REPO / "exports/standalone-middle-576/loop"
        cmd = [
            sys.executable,
            "-m",
            "scripts.middle_depth_long_fit",
            "fit",
            "--arm",
            "loop",
            "--policy",
            str(policy),
        ]
        subprocess.run(
            cmd, cwd=root, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True
        )
        result = read(records / "long-fit-loop.json")
        ref = read(EVIDENCE / "long-fit-loop.json")
        current = read(KIT / "resume.json")["latest"]
        replayed = read(root / TRAIN / current / "state.json")
        with ZipFile(KIT / "resume.zip") as archive:
            original = json.loads(archive.read(str(TRAIN / current / "state.json")))
            opt = TRAIN / current / "optimizer.safetensors"
            expected = load(archive.read(str(opt)))
        actual = load((root / opt).read_bytes())
        if set(actual) != set(expected):
            raise ValueError("Restored optimizer has different named state")

        def identify(tensor):
            return {
                "sha": sha256(tensor.tobytes()).hexdigest(),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }

        tensors = {
            name: {
                "expected": identify(expected[name]),
                "actual": identify(actual[name]),
            }
            for name in sorted(expected)
        }
        tensor_match = all(v["actual"] == v["expected"] for v in tensors.values())
        match = (
            all(
                replayed["hashes"][name] == original["hashes"][name]
                for name in ("adapters.safetensors", "registration.json")
            )
            and tensor_match
        )
        keys = ("step", "gradient", "loss")
        metrics = {k: replayed["log"][-1][k] == original["log"][-1][k] for k in keys}
        features = digest(records / "long-features-loop.json")
        reference_features = read(KIT / "resume.json")["features"]
        passed = (
            match
            and all(metrics.values())
            and result["weights"] == ref["weights"]
            and features == reference_features
        )
        save(
            EVIDENCE / "full-fit-resume-replay.json",
            {
                "passed": passed,
                "archive": digest(KIT / "resume.zip"),
                "step": replayed["step"],
                "hashes": replayed["hashes"],
                "expected": original["hashes"],
                "optimizer_tensors": tensors,
                "optimizer_file_identical": replayed["hashes"][opt.name]
                == original["hashes"][opt.name],
                "identical_metrics": metrics,
                "features": features,
                "reference_features": reference_features,
                "seconds": result["log"][-1]["seconds"],
                "source": "Original MLX trainer restarted from restored step 15 in an independent temporary working directory using four archived full-document features.",
            },
        )
        if not passed:
            raise ValueError("Restored next update differed from the recorded update")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    actions.add_parser("pack")
    actions.add_parser("replay")
    sub = actions.add_parser("restore")
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--previous-only", action="store_true")
    args = p.parse_args()
    if args.action == "pack":
        pack()
    elif args.action == "restore":
        restore(args.out, args.previous_only)
    else:
        replay()
