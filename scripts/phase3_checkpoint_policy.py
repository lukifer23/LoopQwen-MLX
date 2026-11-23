"""Record an execution-only policy transition in a new immutable generation."""

import fcntl
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import read, save, checkpoint_location, digest


def apply(plan):
    root = Path("checkpoints") / ("p3_" + plan["arm"])
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = checkpoint_location(root)
        state = read(parent / "state.json")
        if any(e["plan"] == plan for e in state.get("execution_events", [])):
            return
        if state["step"] != plan["step"] or state["config"]["name"] != root.name:
            raise ValueError("Policy change must occur at its registered boundary")
        state.setdefault("execution_events", []).append(
            dict(
                plan=plan,
                parent=parent.name,
                parent_metadata=digest(parent / "state.json"),
                targets=state["loss_tokens"],
            )
        )
        state["checkpointing"] = plan["checkpointing"]
        tmp = Path(tempfile.mkdtemp(dir=root, prefix=".policy-"))
        try:
            for name in ("adapters.safetensors", "optimizer.safetensors"):
                os.link(parent / name, tmp / name)
            save(tmp / "state.json", state)
            generation = f"step-{state['step']:06d}-{tmp.name[1:]}"
            tmp.rename(root / generation)
            save(root / "latest.json", {"directory": generation})
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)


if __name__ == "__main__":
    apply(
        read(sys.argv[1] if len(sys.argv) == 2 else "experiments/phase3_execution.json")
    )
