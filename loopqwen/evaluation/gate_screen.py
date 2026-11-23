"""Select the separate gate LR after a matched eight-update tuning window."""

from pathlib import Path

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..io import save
from ..training.losses import validation_nll


def score(arm):
    model, _, state = load_trained(Path("checkpoints") / f"p3_{arm}")
    if state["step"] != 8:
        raise ValueError("Expected the eight-update gate LR checkpoint")
    h = state["history"]
    if not any(r["gate_gradient"] and r["gate_delta"] for r in h):
        raise AssertionError("Inactive gate")
    return dict(
        loss=validation_nll(model, rows("validation", "trace")[:30]),
        gate=h[-1]["alpha"],
        gate_logit=h[-1]["gate_logit"],
        gate_lr=h[-1]["gate_lr"],
        order=state["order_hash"],
    )


if __name__ == "__main__":
    import sys

    arm = sys.argv[1]
    save(Path("outputs/phase3") / f"{arm}-lr.json", score(arm))
