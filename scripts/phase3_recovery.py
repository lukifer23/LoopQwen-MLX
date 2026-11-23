"""Exercise explicit fork, native gate moments, strict rejection and SIGTERM recovery."""

import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import read, save, checkpoint_location


def command(args):
    return [sys.executable, "-m", "loopqwen", *args]


def run():
    base = Path("checkpoints/p3_learn1")
    parent = base / read(base / "targets-25000.json")["directory"]
    initial = read(parent / "state.json")
    raw = yaml.safe_load(Path("outputs/phase3/configs/p3_learn1.yaml").read_text())
    paths = {}
    out = Path("outputs/phase3")
    for mode in ("extend", "weights"):
        name = f"p3_audit_{mode}"
        cfg = out / (name + ".yaml")
        cfg.write_text(yaml.safe_dump({**raw, "name": name, "checkpoint_interval": 1}))
        root = Path("checkpoints") / name
        if not root.exists():
            subprocess.run(
                command(
                    [
                        "fork",
                        "--checkpoint",
                        str(parent),
                        "--config",
                        str(cfg),
                        "--mode",
                        mode,
                    ]
                ),
                check=True,
            )
        paths[mode] = root
        state = read(checkpoint_location(root) / "state.json")
        if state["lineage"]["source"] != str(parent):
            raise ValueError("Diagnostic root belongs to another fork")
        if mode == "weights" and (state["step"] or state["cursor"] or state["history"]):
            raise AssertionError("Weights-only initial state carried parent progress")
    root = paths["extend"]
    cfg = out / "p3_audit_extend.yaml"
    if read(checkpoint_location(root) / "state.json")["step"] == initial["step"]:
        proc = subprocess.Popen(
            command(
                ["train", "--config", str(cfg), "--until", str(initial["step"] + 25)]
            )
        )
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError("Worker exited before interruption")
                if (
                    read(checkpoint_location(root) / "state.json")["step"]
                    > initial["step"]
                ):
                    proc.send_signal(signal.SIGTERM)
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("No full update in the diagnostic window")
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=60)
    stopped = read(checkpoint_location(root) / "state.json")
    # SIGTERM can arrive after the immutable generation is published. The writer
    # records cancellation in the run lifecycle even at that boundary.
    lifecycle = read(root / "lifecycle.json")
    if lifecycle["status"] != "interrupted" or not lifecycle.get("resumable"):
        raise AssertionError("Expected interrupted lifecycle")
    subprocess.run(
        command(["train", "--config", str(cfg), "--until", str(stopped["step"] + 1)]),
        check=True,
    )
    restored = read(checkpoint_location(root) / "state.json")
    if (
        restored["step"] != stopped["step"] + 1
        or restored["cursor"] != stopped["cursor"] + 4
    ):
        raise AssertionError("Invalid cursor on controlled resume")
    # Isolate negative restore in a fresh fork. Only mutate the diagnostic copy of metadata.
    bad = "p3_audit_identity"
    badcfg = out / (bad + ".yaml")
    badcfg.write_text(yaml.safe_dump({**raw, "name": bad}))
    dest = Path("checkpoints") / bad
    if not dest.exists():
        subprocess.run(
            command(
                [
                    "fork",
                    "--checkpoint",
                    str(parent),
                    "--config",
                    str(badcfg),
                    "--mode",
                    "extend",
                ]
            ),
            check=True,
        )
    file = checkpoint_location(dest) / "state.json"
    wrong = read(file)
    wrong["training_identity"]["schema"] = -1
    save(file, wrong)
    proc = subprocess.run(
        command(
            ["train", "--config", str(badcfg), "--until", str(initial["step"] + 1)]
        ),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 or "training identity changed" not in proc.stderr:
        raise AssertionError(proc.stderr[-500:])
    save(
        out / "fork-audit.json",
        dict(
            parent=initial["step"],
            interrupted=stopped["step"],
            resumed=restored["step"],
            archived_parent_unchanged=read(parent / "state.json") == initial,
            rejected_incompatible_identity=True,
            weights_only_step=0,
        ),
    )


if __name__ == "__main__":
    run()
