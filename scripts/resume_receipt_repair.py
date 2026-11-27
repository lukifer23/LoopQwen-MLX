"""Check the paused fit or explicitly resume it under its remaining native time budget."""

import argparse
import math
from pathlib import Path
import signal

from loopqwen.io import digest, read
from loopqwen.conditional_reentry import PACKED
from scripts.receipt_repair_fit import measure, plan, PLAN, OUT, PARENT

PAUSE = Path("docs/measurements/receipt-repair-streamed-pause.json")


def run(action):
    s = read(PLAN)
    paused = read(PAUSE)
    if plan() != s or paused["plan"] != digest(PLAN):
        raise ValueError(
            "Training source or implementation differs from the paused run"
        )
    cfg = read(PARENT / "registration.json")
    fit = read(PARENT / "fit.json")
    directory = Path(cfg["checkpoint"])
    expected = {
        PARENT / "step.safetensors": fit["weights"],
        directory / "config.json": cfg["config_sha256"],
        directory.parent
        / "conditional-reentry-backbone"
        / PACKED: fit["backbone"]["sha256"],
    }
    for path, value in expected.items():
        if not path.is_file() or digest(path) != value:
            raise ValueError(f"Required asset missing or different: {path}")
    seconds = s["seconds"] - math.ceil(paused["elapsed_seconds"])
    if seconds <= 0:
        raise ValueError("Registered time exhausted")
    if (OUT / "receipt.json").exists():
        print("Terminal decision exists:", OUT / "receipt.json")
        return
    print(
        f"Verified source, parent and donor. {seconds} native seconds remain.",
        flush=True,
    )
    if action == "fit":

        def stop(_signal, _frame):
            raise InterruptedError("Remaining registered time exhausted")

        signal.signal(signal.SIGALRM, stop)
        signal.alarm(seconds)
        try:
            measure()
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "fit"))
    run(parser.parse_args().action)
