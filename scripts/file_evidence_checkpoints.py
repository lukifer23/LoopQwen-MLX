"""Verify the recovered native training state and publish its final AdamW moments."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open

from loopqwen.io import digest, read, save
from scripts.file_evidence_audit import NAME


def shape_signature(path):
    with safe_open(path, framework="np") as tensors:
        return {k: tuple(tensors.get_slice(k).get_shape()) for k in tensors.keys()}


def verify(root):
    s = read(root / "registration.json")
    fit = read(root / "fit.json")
    environment = read(root / "environment.json")
    assert environment["registration_sha256"] == digest(root / "registration.json")
    for path, expected in environment["execution_sources"].items():
        source = subprocess.check_output(
            ["git", "show", environment["execution_revision"] + ":" + path]
        )
        assert hashlib.sha256(source).hexdigest() == expected
    history = read(root / "optimizer-steps.json")
    steps = list(
        range(s["checkpoint_interval"], s["updates"], s["checkpoint_interval"])
    ) + [s["updates"]]
    assert [h["step"] for h in history] == steps
    assert all(
        0 < a["targets"] < b["targets"]
        and 0 < a["optimizer_seconds"] < b["optimizer_seconds"]
        for a, b in zip(history, history[1:])
    )
    assert history[-1]["targets"] == fit["targets"]
    optimizer = root / "optimizer.safetensors"
    assert digest(optimizer) == history[-1]["optimizer_sha256"]
    weights = root / "tool-adapters.safetensors"
    assert digest(weights) == fit["weights"]
    adapter_shapes = shape_signature(weights)
    assert (
        set(adapter_shapes)
        and sum(int(np.prod(s)) for s in adapter_shapes.values()) == fit["trainable"]
    )
    moments = {
        k + "." + suffix: shape
        for k, shape in adapter_shapes.items()
        for suffix in ("m", "v")
    }
    assert shape_signature(optimizer) == {**moments, "step": (), "learning_rate": ()}
    with safe_open(optimizer, framework="np") as opt:
        assert int(opt.get_tensor("step").item()) == fit["updates"]
        assert np.isclose(opt.get_tensor("learning_rate").item(), s["lr"])
        for key in moments:
            data = opt.get_tensor(key)
            assert np.isfinite(data).all() and (
                not key.endswith(".v") or (data >= 0).all()
            )
    with safe_open(weights, framework="np") as w:
        assert all(np.isfinite(w.get_tensor(k)).all() for k in w.keys())
    print(
        len(history),
        "optimizer checkpoints;",
        len(moments),
        "finite final moment tensors;",
        fit["targets"],
        "targets",
    )


def record(root):
    spec = read(root / "registration.json")
    fit = read(root / "fit.json")
    checkpoints = Path("checkpoints/research") / spec["id"]
    end = checkpoints / read(checkpoints / "latest.json")["directory"]
    history = []
    for directory in sorted(checkpoints.glob("step-*")):
        state = read(directory / "state.json")
        assert state["registration"] == digest(root / "registration.json")
        assert state["parity"] == read(root / "gradient-parity.json")
        assert state["loss"] == [l for l in fit["loss"] if l["step"] <= state["step"]]
        history.append(
            {
                **{k: state[k] for k in ("step", "targets", "optimizer_seconds")},
                "weight_sha256": digest(directory / "adapters.safetensors"),
                "optimizer_sha256": digest(directory / "optimizer.safetensors"),
            }
        )
    with (
        safe_open(end / "adapters.safetensors", framework="np") as a,
        safe_open(root / "tool-adapters.safetensors", framework="np") as b,
    ):
        assert a.keys() == b.keys() and all(
            np.array_equal(a.get_tensor(k), b.get_tensor(k)) for k in a.keys()
        )
    dest = root / "optimizer.safetensors"
    if dest.exists():
        assert digest(dest) == history[-1]["optimizer_sha256"]
    else:
        shutil.copyfile(end / "optimizer.safetensors", dest.with_suffix(".tmp"))
        dest.with_suffix(".tmp").replace(dest)
    if (root / "optimizer-steps.json").exists():
        assert read(root / "optimizer-steps.json") == history
    else:
        save(root / "optimizer-steps.json", history)
    verify(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    base = "outputs/research" if args.record else "research/results"
    (record if args.record else verify)(Path(base) / NAME)
