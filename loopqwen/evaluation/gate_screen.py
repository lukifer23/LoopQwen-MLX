"""Select the separate gate LR after a matched eight-update tuning window."""

from pathlib import Path

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..io import save, read, digest
from ..identity import fingerprint
from .identity import evaluation_identity
from ..training.losses import validation_nll


def score(arm):
    root = Path("checkpoints") / f"p3_{arm}"
    generations = list(root.glob("step-000008-*"))
    if len(generations) != 1:
        raise ValueError("Need a unique immutable eight-update checkpoint")
    generation = generations[0]
    samples = rows("validation", "trace")[:30]
    identity = evaluation_identity(
        {
            "study": read("experiments/phase3_dev.json"),
            "measurement_code": digest(Path(__file__)),
        },
        generation,
        [r["id"] for r in samples],
        {"protocol": "teacher-forced", "batch_size": 4},
        "token-weighted-nll",
    )
    dest = Path("outputs/phase3/gate-lr") / arm / (fingerprint(identity) + ".json")
    if dest.exists():
        record = read(dest)
        if record["identity"] != identity:
            raise ValueError("Gate LR measurement identity mismatch")
        return record
    model, _, state = load_trained(generation)
    if state["step"] != 8:
        raise ValueError("Expected the eight-update gate LR checkpoint")
    h = state["history"]
    if not any(r["gate_gradient"] and r["gate_delta"] for r in h):
        raise AssertionError("Inactive gate")
    result = dict(
        identity=identity,
        loss=validation_nll(model, samples),
        gate=h[-1]["alpha"],
        gate_logit=h[-1]["gate_logit"],
        gate_lr=h[-1]["gate_lr"],
        order=state["order_hash"],
    )

    save(dest, result)
    return result


if __name__ == "__main__":
    import sys

    arm = sys.argv[1]
    result = score(arm)
    folder = Path("outputs/phase3")
    save(folder / f"{arm}-lr-current.json", result)
    if not (folder / f"{arm}-lr.json").exists():
        save(folder / f"{arm}-lr.json", result)
