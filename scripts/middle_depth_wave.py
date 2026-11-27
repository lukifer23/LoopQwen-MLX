"""Run resumable first-wave decisions in source-counterbalanced order on one GPU."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

from loopqwen.io import read
from scripts.middle_depth_budget import ROOT


def dispatch(module, *args):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.middle_depth_" + module,
        *map(str, args),
    ]
    failures = 0
    while True:
        while (
            subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            time.sleep(10)
        begin = time.time()
        run = subprocess.run(cmd)
        if run.returncode == 0:
            return
        charges = [
            r
            for p in (ROOT / "charges").glob("*.json")
            if (r := read(p))["dispatch"] >= begin
        ]
        if len(charges) != 1:
            raise RuntimeError(f"Inspect worker failure for {cmd}")
        r = charges[0]
        last = r.get("last", {})
        if (
            "Native development limit" not in r.get("reason", "")
            or last.get("pressure", 1) == 1
            or failures >= 4
        ):
            raise RuntimeError(f"Stopped {cmd}, {r.get('reason')}")
        failures += 1
        time.sleep(15)


def fit(arm, step):
    folder = Path("checkpoints/middle-depth/4619") / arm
    state = read(folder / read(folder / "latest.json")["path"] / "state.json")
    if state["step"] < step:
        dispatch("fit", arm, "--until", step)
    return folder / f"step-{step:04d}"


def generate(stage, arms, budgets, policies):
    for tokens in budgets:
        for block in (0, 1):
            for arm in arms if block == 0 else reversed(arms):
                # collect handles row-level restarts and validates existing identities.
                args = [
                    "collect",
                    "--arm",
                    arm,
                    "--stage",
                    stage,
                    "--block",
                    block,
                    "--tokens",
                    tokens,
                ]
                if arm in policies:
                    args += ["--policy", policies[arm]]
                dispatch("development", *args)


def first():
    early = {arm: fit(arm, 32) for arm in ("loop", "ordinary")}
    generate("initial", ["loop", "base"], [128], {})
    generate("early", ["ordinary", "loop"], [128], early)
    for arm, path in early.items():
        dispatch("histories", "collect", "--arm", arm, "--policy", path)
    if (
        not (ROOT / "sparse-parity.json").is_file()
        or not read(ROOT / "sparse-parity.json")["passed"]
    ):
        dispatch("verify")
    dispatch(
        "documents", "run", "--arm", "base", "--book", "book", "--world", "original"
    )


def paired_fit():
    # Keep a common usable endpoint after each epoch, even on a resource stop.
    for epoch, step in enumerate((144, 288, 432, 576)):
        arms = ("ordinary", "loop") if epoch % 2 == 0 else ("loop", "ordinary")
        for arm in arms:
            fit(arm, step)


def short_terminal():
    policies = {arm: fit(arm, 576) for arm in ("loop", "ordinary")}
    generate("terminal", ["loop", "ordinary", "base", "instruction"], [128, 512], policies)


def depths():
    policy = fit("loop", 576)
    for block in (0, 1):
        for depth in ((1, 2, 3) if block == 0 else (3, 2, 1)):
            dispatch("development", "collect", "--stage", "depth", "--arm", "loop",
                     "--depth", depth, "--policy", policy, "--tokens", 512, "--block", block)


if __name__ == "__main__":
    actions = {"first": first, "fit": paired_fit, "short-terminal": short_terminal, "depth": depths}
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=actions)
    actions[p.parse_args().stage]()
