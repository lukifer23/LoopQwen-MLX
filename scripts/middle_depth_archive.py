"""Retain two relocatable policies against one content-addressed certified donor."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from loopqwen.io import digest, read, save
from loopqwen.ordinary_model import CERTIFIED_INVENTORY
from scripts.middle_depth_budget import BASE, ROOT
from scripts.verify_runtime_wheel import build

ARCHIVE = Path("exports/middle-depth")
DOCUMENTS = (
    "config.json",
    "provenance.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tool/tool-model.json",
    "LICENSE",
)


def retain(src, view, objects):
    sha = digest(src)
    blob = objects / sha
    if not blob.exists():
        with tempfile.TemporaryDirectory(dir=objects) as tmp:
            path = Path(tmp) / sha
            # Copy-on-write on APFS. Existing releases keep independent files.
            r = subprocess.run(["cp", "-c", str(src), str(path)], capture_output=True)
            if r.returncode:
                shutil.copyfile(src, path)
            if digest(path) != sha:
                raise ValueError("Incorrect retained bytes")
            path.replace(blob)
    if digest(blob) != sha:
        raise ValueError("Damaged archive object")
    view.parent.mkdir(exist_ok=True, parents=True)
    link = os.path.relpath(blob, view.parent)
    if view.is_symlink():
        if os.readlink(view) != link:
            raise ValueError(f"Existing archive view differs at {view}")
    elif view.exists():
        raise FileExistsError(view)
    else:
        view.symlink_to(link)
    return dict(sha256=sha, bytes=src.stat().st_size)


def archive(stage, loop, ordinary):
    objects = ARCHIVE / "objects"
    objects.mkdir(exist_ok=True, parents=True)
    bundle = ARCHIVE / "models" / stage
    data = {}
    inv = BASE / "tool-export-files.json"
    if digest(inv) != CERTIFIED_INVENTORY:
        raise ValueError("Wrong certified donor")
    members = read(inv)
    names = [k for k in DOCUMENTS if k in members] + [
        k for k in members if k.startswith("chat_templates/")
    ]
    for name in names + ["tool-export-files.json"]:
        f = BASE / name
        if name in members and members[name] != digest(f):
            raise ValueError(name)
        data["donor/" + name] = retain(f, ARCHIVE / "donor" / name, objects)
    single = sum(item["bytes"] for item in data.values())
    for arm, src in (("loop", loop), ("ordinary", ordinary)):
        spec = read(src / "registration.json")
        if (
            spec["weights"] != digest(src / "adapters.safetensors")
            or spec["donor"] != "base"
        ):
            raise ValueError("Invalid selected policy")
        for name in ("registration.json", "adapters.safetensors"):
            view = bundle / arm / name
            data[str(view.relative_to(ARCHIVE))] = retain(src / name, view, objects)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory() as tmp:
        wheel = build(Path(tmp), revision)
        view = ARCHIVE / "wheels" / stage / wheel.name
        data[str(view.relative_to(ARCHIVE))] = retain(wheel, view, objects)
    unique = {r["sha256"]: r["bytes"] for r in data.values()}
    report = dict(
        stage=stage,
        revision=revision,
        files=data,
        unique_payload_bytes=sum(unique.values()),
        two_model_payload_bytes=sum(x["bytes"] for x in data.values()) + single,
        shared_donor_bytes=single,
        retention="Relative view links resolve to hash-checked archive objects. Export paths and original certified release are intact. Counts are logical payload bytes; APFS extent ownership can differ.",
    )
    save(ARCHIVE / "manifests" / (stage + ".json"), report)
    save(ROOT / ("archive-" + stage + ".json"), report)


def verify(directory, stage):
    cfg = read(directory / "manifests" / (stage + ".json"))
    for name, item in cfg["files"].items():
        p = directory / name
        blob = directory / "objects" / item["sha256"]
        if (
            p.resolve() != blob.resolve()
            or digest(blob) != item["sha256"]
            or p.stat().st_size != item["bytes"]
        ):
            raise ValueError(f"Incomplete relocatable archive at {name}")
    print(
        len(cfg["files"]),
        "verified archive members",
        cfg["shared_donor_bytes"],
        "shared donor bytes",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("build", "verify"))
    p.add_argument("--directory", type=Path, default=ARCHIVE)
    p.add_argument("--stage", required=True)
    p.add_argument("--loop", type=Path)
    p.add_argument("--ordinary", type=Path)
    a = p.parse_args()
    if a.action == "build":
        archive(a.stage, a.loop, a.ordinary)
    verify(a.directory, a.stage)
