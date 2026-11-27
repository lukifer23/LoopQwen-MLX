"""Run one registered native source-repair stage and release its process after each pressure warning."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from loopqwen.io import read, save
from scripts.register_repair_query import FEATURES, FIT, PRIMARY

MODULE = {
    FEATURES: "shared_query_features",
    FIT: "shared_query_fit",
    PRIMARY: "shared_query_transfer",
}


def work(name):
    spec = read(Path("research/experiments") / (name + ".json"))
    session = read("outputs/research/active-session.json")
    root = Path("outputs/research") / name
    deadline = (
        session["start"] + read("research/state.json")["limits"]["wall_seconds"] - 20
    )
    normal = None
    while time.time() < deadline:
        pressure = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
        ).strip()
        if (
            pressure != "1"
            or psutil.virtual_memory().available
            < spec["limits"]["available_gib"] * 2**30
        ):
            normal = None
        elif normal is None:
            normal = time.monotonic()
        elif time.monotonic() - normal >= 10:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    f"from loopqwen.research.{MODULE[name]} import run; run({name!r})",
                ],
                check=True,
            )
            if (root / "decision.json").exists():
                return
            if (
                not spec.get("max_pressure_restarts")
                or not (root / "pressure-stops.json").exists()
            ):
                raise RuntimeError(
                    "A native stage returned without a decision or pressure record"
                )
            normal = None
        time.sleep(2)
    root.mkdir(exist_ok=True, parents=True)
    save(root / "registration.json", spec)
    decision = {
        "passed": False,
        "status": "interrupted",
        "reason": "Host availability window expired",
    }
    if name == PRIMARY:
        decision["durable"] = {}
    elif name == FEATURES:
        from loopqwen.evaluation.mechanism import unpack
        from loopqwen.research.conditional_features import feature_root

        journal = feature_root(spec).parent / "partial-index.json.gz"
        if journal.exists():
            shutil.copyfile(journal, root / "partial-index.json.gz")
            decision["durable"] = len(unpack(journal))
    elif (root / "checkpoint.json").is_file():
        from loopqwen.io import digest

        folder = Path(read(root / "checkpoint.json")["directory"])
        snapshot = read(folder / "state.json")
        partial = {}
        for filename in ("state.json", "adapters.safetensors", "optimizer.safetensors"):
            dest = root / ("partial-" + filename)
            shutil.copyfile(folder / filename, dest)
            partial[dest.name] = digest(dest)
        decision.update(
            partial_hashes=partial,
            durable_updates=snapshot["step"],
            durable_labels=sum(r["positions"] for r in snapshot["history"]),
        )
        (root / "checkpoint.json").unlink()
    save(root / "decision.json", decision)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=MODULE)
    work(p.parse_args().stage)
