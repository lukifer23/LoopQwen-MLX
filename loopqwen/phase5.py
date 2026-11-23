"""Standalone native pointer study. All operations after the data audit use a single GPU lease."""

import argparse
import subprocess
import sys
from .data.pointer5 import build
from .io import read
from .training.pointer5 import SPEC, OUT, install, profile
from .evaluation.pointer5 import installed, milestone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stages",
        nargs="+",
        choices=[
            "audit",
            "data",
            "profile",
            "smoke",
            "install",
            "evaluate",
            "diagnostics",
            "followup",
        ],
    )
    p.add_argument("--seed", type=int, default=SPEC["seed"])
    args = p.parse_args()
    for stage in args.stages:
        if stage == "audit":
            subprocess.run(
                [sys.executable, "scripts/verify_phase4_archive.py"], check=True
            )
        elif stage == "data":
            print(build()["files"])
        elif stage == "profile":
            profile(args.seed)
        elif stage in ("smoke", "install"):
            if not (OUT / "profile.json").exists():
                profile(args.seed)
            for arm in SPEC["arms"]:
                install(args.seed, arm, 24 if stage == "smoke" else 480)
        elif stage == "evaluate":
            for arm in SPEC["arms"]:
                milestone(args.seed, arm, 480)
        elif stage == "diagnostics":
            from .evaluation.pointer5_diagnostics import diagnose

            diagnose(args.seed, "recurrent")
        elif stage == "followup":
            from .evaluation.pointer5_diagnostics import followup

            followup(args.seed)


if __name__ == "__main__":
    main()
