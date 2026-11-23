"""Fork a strict run into a new namespace, preserving the immutable parent."""

from copy import deepcopy
from pathlib import Path
import tempfile
import shutil

import mlx.core as mx
from mlx.utils import tree_flatten

from .. import checkpoint
from ..config import config
from ..data.sampling import prepare
from ..identity import training_identity
from ..io import digest, read
from .optimizers import optimizer


def fork(source, path, mode, migration_reason=None):
    if mode not in ("extend", "weights", "compatible"):
        raise ValueError("Specify extend, weights or compatible")
    c = config(path)
    if not c.get("strict_resume"):
        raise ValueError("Fork destination must use strict_resume")
    generation = checkpoint.location(source)
    parent = read(generation / "state.json")
    dest = Path("checkpoints") / c["name"]
    if dest.exists():
        raise FileExistsError(dest)
    _, _, manifest, sampling = prepare(c)
    current = training_identity(c, manifest)
    if mode == "extend":
        allowed = {
            "name",
            "label_budget",
            "checkpoint_interval",
            "eval_interval",
            "eval_count",
            "milestones",
        }
        if (
            c.get("schedule", "constant") == "constant"
            and "warmup_fraction" not in c
            and c["curriculum"] == "direct"
            and c["alpha_schedule"] is None
        ):
            allowed.add("steps")
        original = parent["config"]
        if any(
            original.get(k) != c.get(k) for k in (original.keys() | c.keys()) - allowed
        ):
            raise ValueError(
                "Extension changes the optimization, data or recurrence policy"
            )
        if (
            c["steps"] <= parent["step"]
            or c.get("label_budget", float("inf")) <= parent["loss_tokens"]
        ):
            raise ValueError("Extended budget must allow another full update")
        if parent["manifest"] != manifest or parent.get("sampling") != sampling:
            raise ValueError("Data schedule mismatch")
        old_identity = parent.get("training_identity", {})
        if (
            any(current[k] != old_identity.get(k) for k in current if k != "config")
            and not migration_reason
        ):
            raise ValueError(
                "Cross-implementation extension requires an explicit migration reason"
            )
    model, _, info = checkpoint.setup(c)
    opt = optimizer(c)
    transfer = None
    if mode == "compatible":
        if (
            not migration_reason
            or parent["config"]["base_revision"] != c["base_revision"]
        ):
            raise ValueError(
                "Compatible transfer requires the same base and a documented reason"
            )
        incoming = mx.load(str(generation / "adapters.safetensors"))
        expected = dict(tree_flatten(model.trainable_parameters()))
        names = set(expected) & set(incoming)
        mismatch = sorted(k for k in names if expected[k].shape != incoming[k].shape)
        restored = sorted(names - set(mismatch))
        if not restored:
            raise ValueError(f"No compatible parameters. Shape mismatches: {mismatch}")
        transfer = dict(
            restored=restored,
            newly_initialized=sorted(set(expected) - set(restored)),
            omitted=sorted(set(incoming) - set(restored)),
            shape_mismatches=mismatch,
            optimizer="fresh",
        )
        model.load_weights([(k, incoming[k]) for k in restored], strict=False)
    else:
        checkpoint.restore(generation, model, opt if mode == "extend" else None)
    state = deepcopy(parent)
    if mode in ("weights", "compatible"):
        opt.init(model.trainable_parameters())
        mx.eval(opt.state)
        state.update(
            step=0,
            cursor=0,
            history=[],
            visits={},
            input_tokens=0,
            loss_tokens=0,
            order_hash="",
            sequence_length=c["sequence_length"],
            batch_size=c["batch_size"],
            checkpointing=c["gradient_checkpointing"],
            alpha_override=model.alpha_override,
        )
    import subprocess

    state.update(
        config=c,
        model=info,
        trainable=sum(a.size for _, a in tree_flatten(model.trainable_parameters())),
        status="started",
        manifest=manifest,
        sampling=sampling,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        training_identity=current,
        lineage=dict(
            mode=mode,
            source=str(generation),
            metadata=digest(generation / "state.json"),
            weights=digest(generation / "adapters.safetensors"),
            migration_reason=migration_reason,
            transfer=transfer,
        ),
    )
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".fork-"))
    try:
        checkpoint.write(tmp, model, opt, state)
        tmp.rename(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return str(dest)
