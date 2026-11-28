"""Restore the complete trained model from its parent and small hashed generations."""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.io import digest, read, save
from scripts.middle_depth_standalone import MODELS, check
from scripts.middle_read_fit import REPORT, RUN, validate
from scripts.verify_runtime_wheel import build

REVISION = "840e8ad305d5e4faf2ac6296e4f84b90d1de7d70"
DEST = MODELS.parent / "standalone-middle-read-0084"
ARCHIVE = Path("research/results/middle-read-0084")


def retain():
    fit = read(REPORT / "early-fit.json")
    if not read(REPORT / "restart.json")["passed"]:
        raise ValueError("The independent next update differs")
    for number in (72, 84):
        origin = RUN / "fit" / f"step-{number:04d}"
        validate(origin, fit["identity"])
        target = ARCHIVE / origin.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with TemporaryDirectory(dir=target.parent) as tmp:
                new = Path(tmp) / origin.name
                shutil.copytree(origin, new)
                validate(new, fit["identity"])
                new.rename(target)
        if digest(target / "state.json") != digest(origin / "state.json"):
            raise ValueError("Archived optimizer state differs")
        validate(target, fit["identity"])
    if digest(ARCHIVE / "step-0084/state.json") != fit["state"]:
        raise ValueError("Terminal generation differs")


def restore(destination):
    fit = read(REPORT / "early-fit.json")
    state = validate(ARCHIVE / "step-0084", fit["identity"])
    controller = ARCHIVE / "step-0084/controller.safetensors"
    if (
        digest(controller) != fit["weights"]
        or digest(controller.parent / "state.json") != fit["state"]
        or state["step"] != 84
    ):
        raise ValueError("Unregistered update")
    parent = MODELS / "loop"
    manifest = check(parent, "loop")
    if manifest["files"]["adapters.safetensors"] != fit["identity"]["parent"]:
        raise ValueError("Different fitted parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        tmp = Path(tmp)
        wheel = build(tmp.resolve(), REVISION)
        spec = read(parent / "standalone-middle.json")
        policy = tmp / "read-policy.json"
        marker = tmp / "standalone-middle.json"
        save(
            policy,
            {
                "schema": 1,
                "parent": fit["identity"]["parent"],
                "weights": fit["weights"],
                "reads": 2,
                "seed": fit["identity"]["seed"],
                "fit": digest(REPORT / "early-fit.json"),
            },
        )
        save(marker, dict(spec, schema=2, read_policy_sha256=digest(policy)))
        replacements = [wheel, policy, marker, controller]
        files = dict(manifest["files"])
        files.update({p.name: digest(p) for p in replacements})
        full = {
            "files": files,
            "revision": REVISION,
            "parent_manifest": digest(
                Path("docs/measurements/middle-depth/standalone/loop-files.json")
            ),
            "state": fit["state"],
        }
        record = REPORT / "standalone.json"
        if record.exists() and read(record) != full:
            raise ValueError("Rebuilt standalone file identities differ")
        if not destination.exists():
            fresh = tmp / "model"
            subprocess.run(["cp", "-cR", parent, fresh], check=True)
            for p in replacements:
                shutil.copyfile(p, fresh / p.name)
            check(fresh, "loop", full)
            fresh.rename(destination)
        check(destination, "loop", full)
        save(record, full)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("retain", "restore"))
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    if args.action == "retain":
        retain()
    print(restore(args.destination))
