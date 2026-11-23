"""Materialize a common training schedule and launch one sequential worker at a time."""

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import digest, read, save


def materialize(arm, policy="protocol_and_length", ckpt="on", diagnostic=False):
    from loopqwen.config import config
    from loopqwen.data.sampling import prepare

    study = read("experiments/phase3_dev.json")
    if digest(Path("data/v2/manifest.json")) != study["data_sha256"]:
        raise ValueError("Dataset registration mismatch")
    name = "p3_" + arm + (f"_profile_{policy}_{ckpt}" if diagnostic else "")
    c = dict(
        name=name,
        dataset="v2",
        mixture="mixed_answer_and_traces",
        sequence_length=256,
        batch_size=4,
        accumulation=1,
        steps=6000,
        warmup=5,
        checkpoint_interval=150,
        eval_interval=6000,
        eval_count=30,
        validation_batch_size=4,
        memory_target_gib=8,
        min_available_gib=1,
        gradient_checkpointing=ckpt != "off",
        checkpoint_policy="block" if ckpt == "block" else "all",
        seed=study["seed"],
        strict_resume=True,
        bucket_policy=policy,
        label_budget=250000,
        milestones=study["milestones"],
        **study["arms"][arm],
    )
    path = Path("outputs/phase3/configs") / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(c)
    if path.exists() and path.read_text() != body:
        raise ValueError("Existing run config differs")
    path.write_text(body)
    rows, _, _, _ = prepare(config(path))
    targets = 0
    steps = {}
    h = hashlib.sha256()
    for j in range(c["steps"]):
        batch = [rows[(4 * j + i) % len(rows)] for i in range(4)]
        h.update(
            json.dumps(
                [(r["id"], r["protocol"], r["prefix"], r["ids"]) for r in batch],
                separators=(",", ":"),
            ).encode()
        )
        targets += sum(r["output_length"] for r in batch)
        for m in study["milestones"]:
            if targets >= m and str(m) not in steps:
                steps[str(m)] = dict(
                    step=j + 1, targets=targets, schedule=h.hexdigest()
                )
        if targets >= c["label_budget"]:
            break
    save(path.with_suffix(".schedule.json"), steps)
    return path, steps


def run(arm, targets=None, updates=None, **kw):
    path, plan = materialize(arm, **kw)
    stop = updates or plan[str(targets)]["step"]
    root = Path("checkpoints") / path.stem
    if (root / "latest.json").exists():
        state = read(root / read(root / "latest.json")["directory"] / "state.json")
        if state["step"] > stop:
            return
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "loopqwen",
            "train",
            "--config",
            str(path),
            "--until",
            str(stop),
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("arm", choices=read("experiments/phase3_dev.json")["arms"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--targets", type=int, choices=(10000, 25000, 50000, 100000, 250000))
    g.add_argument("--updates", type=int)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--policy", default="protocol_and_length")
    p.add_argument("--checkpointing", choices=("on", "off", "block"), default="on")
    a = p.parse_args()
    run(
        a.arm,
        a.targets,
        a.updates,
        policy=a.policy,
        ckpt=a.checkpointing,
        diagnostic=a.profile,
    )
