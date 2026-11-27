"""Release the native process on host pressure and bound wall time and all native attempts."""

import subprocess
import sys
import time
from pathlib import Path

import psutil

from loopqwen.io import read, save
from scripts.register_elastic_world import FIT


def run():
    root = Path("outputs/research") / FIT
    spec = read(Path("research/experiments") / (FIT + ".json"))
    state = read("research/state.json")
    start = read("outputs/research/active-session.json")["start"]
    deadline = start + state["limits"]["wall_seconds"] - 25
    nominal = None
    while time.time() < deadline:
        pressure = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
        ).strip()
        if (
            pressure != "1"
            or psutil.virtual_memory().available
            < spec["limits"]["available_gib"] * 2**30
        ):
            nominal = None
        elif nominal is None:
            nominal = time.monotonic()
        elif time.monotonic() - nominal >= 10:
            job = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    f"from loopqwen.research.contrastive_fit import run; run({FIT!r})",
                ]
            )
            if (root / "decision.json").is_file():
                job.check_returncode()
                return
            job.check_returncode()
            nominal = None
        time.sleep(2)
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    save(
        root / "decision.json",
        {
            "passed": False,
            "status": "interrupted",
            "reason": "Two-hour host availability window expired",
        },
    )


if __name__ == "__main__":
    run()
