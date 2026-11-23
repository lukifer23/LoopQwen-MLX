"""Check the effective, currently used coefficients of fixed and scheduled controls."""

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import read, save, checkpoint_location


def run():
    out = Path("outputs/phase3")
    cfg = yaml.safe_load((out / "configs/p3_fixed.yaml").read_text())
    cfg.pop("fixed_alpha")
    cfg.update(
        name="p3_schedule_audit",
        steps=3,
        alpha_schedule=[0.0, 0.1],
        warmup=1,
        checkpoint_interval=3,
        eval_interval=3,
        eval_count=4,
        milestones=[],
    )
    path = out / "p3_schedule_audit.yaml"
    path.write_text(yaml.safe_dump(cfg))
    subprocess.run(
        [sys.executable, "-m", "loopqwen", "train", "--config", str(path)], check=True
    )
    state = read(checkpoint_location("checkpoints/p3_schedule_audit") / "state.json")
    values = [r["alpha"] for r in state["history"]]
    if (
        values != [0.0, 0.05, 0.1]
        or state["trainable"] != 1336320
        or any("gate_gradient" in r for r in state["history"])
    ):
        raise AssertionError(values)
    fixed = read(checkpoint_location("checkpoints/p3_fixed") / "state.json")
    if fixed["trainable"] != 1336320 or {r["alpha"] for r in fixed["history"]} != {0.1}:
        raise AssertionError("Fixed coefficient telemetry or count")
    save(
        out / "gate-controls.json",
        dict(scheduled=values, fixed=0.1, active_gate_parameters=0),
    )


if __name__ == "__main__":
    run()
