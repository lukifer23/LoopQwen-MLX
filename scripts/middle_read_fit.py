"""Fit both exits of the middle evidence read, resuming only hash-checked native updates."""

import argparse
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import dependencies
from loopqwen.io import digest, read, save
from loopqwen.middle_read import LAYER, install, set_reads
from loopqwen.tool_cli import open_assistant
from loopqwen.training.optimizers import optimizer
from scripts.middle_depth_budget import worker
from scripts.middle_depth_fit import tensors
from scripts.middle_depth_standalone import MODELS
from scripts.middle_read_material import BANK, SPEC
from scripts.middle_read_profile import PLAN, REPORT

RUN = Path("checkpoints/middle-read")
EARLY = SPEC.with_name("middle-read-early.md")


def persist(folder, step, opt, number, identity, history):
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"step-{number:04d}"
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=folder) as temporary:
        tmp = Path(temporary) / "generation"
        tmp.mkdir()
        tensors(tmp / "controller.safetensors", step.parameters())
        tensors(tmp / "optimizer.safetensors", opt.state)
        save(
            tmp / "state.json",
            {
                "step": number,
                "next_seed": identity["seed"] + number,
                "identity": identity,
                "log": history,
                "hashes": {p.name: digest(p) for p in tmp.iterdir()},
            },
        )
        os.rename(tmp, dest)
    versions = sorted(folder.glob("step-*/state.json"))
    save(
        folder / "latest.json",
        {
            "latest": dest.name,
            "state": digest(dest / "state.json"),
            "previous": versions[-2].parent.name if len(versions) > 1 else None,
        },
    )


def validate(path, identity=None):
    state = read(path / "state.json")
    if (
        identity is not None
        and state["identity"] != identity
        or path.name != f"step-{state['step']:04d}"
        or state["next_seed"] != state["identity"]["seed"] + state["step"]
        or set(state["hashes"]) != {"optimizer.safetensors", "controller.safetensors"}
        or any(digest(path / file) != sha for file, sha in state["hashes"].items())
    ):
        raise ValueError("Invalid middle-read resume generation")
    return state


def derivative(m, step, r, spec, guard):
    x = mx.array([r["p"] + r["y"][:-1]])
    anchor = mx.stop_gradient(m._execute(m.text.model.embed_tokens(x), m.sites[:LAYER]))
    mx.eval(anchor)
    labels = mx.array(r["y"])

    def objective():
        h = m._execute(anchor, m.sites[LAYER:])
        h = m.text.model.norm(h[:, -labels.size :])
        logits = m.project(h)[0].astype(mx.float32)
        val = nn.losses.cross_entropy(logits, labels).mean()
        if r["negatives"]:
            pos, wrong = zip(*r["negatives"], strict=True)
            k = mx.array(pos)
            val += (
                0.65
                * nn.softplus(
                    1 + logits[k, mx.array(wrong)] - logits[k, labels[k]]
                ).mean()
            )
        return val

    values = []
    grads = []
    for n in spec["depths"]:
        set_reads(m, n)
        value, g = nn.value_and_grad(step, objective)()
        mx.eval(value, g)
        values.append(value.item())
        grads.append(g)
        guard(force=True)
    grad = tree_map(lambda a, b: (a + b) * 0.5, *grads)
    grad, norm = optim.clip_grad_norm(grad, 1.0)
    magnitude = norm.item()
    if not all(map(math.isfinite, [*values, magnitude])):
        raise FloatingPointError("Nonfinite native update")
    return grad, values, magnitude


def fit(until, restart_check=False, pause_prepared=False):
    spec = read(SPEC)
    profile = read(REPORT / "profile.json")
    if (
        not profile["passed"]
        or profile["plan"] != digest(PLAN)
        or spec["data"] != digest(BANK)
    ):
        raise ValueError("Numerical admission or source identity failed")
    if not 1 <= until <= spec["early_step"]:
        raise ValueError(
            "The current admission covers updates through the early decision"
        )
    data = unpack(BANK)
    if [r["id"] for r in data] != spec["order"]:
        raise ValueError("Changed training order")
    with worker(
        "fit", "middle-read/" + ("restart" if restart_check else "fit")
    ) as guard:
        parent = MODELS / "loop"
        a = open_assistant(argparse.Namespace(checkpoint=parent, tool_format="qwen"))
        m = a._tool_model
        step = install(m, 2, spec["seed"]).step
        m.train()
        ident = {
            "plan": digest(PLAN),
            "early_decision": digest(EARLY),
            "spec": digest(SPEC),
            "data": digest(BANK),
            "base": profile["base"],
            "parent": digest(parent / "adapters.safetensors"),
            "seed": spec["seed"],
            "numerics": dependencies(),
            "implementation": {
                str(p): digest(p)
                for p in (
                    Path(__file__).resolve().relative_to(Path.cwd()),
                    Path("loopqwen/middle_read.py"),
                    Path("loopqwen/shared_query.py"),
                    Path("loopqwen/training/optimizers.py"),
                )
            },
        }
        if ident["parent"] != profile["parent"]:
            raise ValueError("Parent weights changed")
        opt = optimizer(
            {
                "learning_rate": spec["lr"],
                "warmup": spec["warmup"],
                "steps": len(data) * spec["epochs_max"],
                "weight_decay": 0.01,
            }
        )
        folder = RUN / "fit"
        versions = sorted(folder.glob("step-*/state.json"))
        history = []
        number = 0
        if versions:
            path = folder / "step-0001" if restart_check else versions[-1].parent
            state = validate(path, ident)
            step.load_weights(str(path / "controller.safetensors"), strict=True)
            opt.state = tree_unflatten(
                list(mx.load(str(path / "optimizer.safetensors")).items())
            )
            number, history = state["step"], state["log"]
        if restart_check and (
            number != 1 or until != 2 or not (folder / "step-0002").exists()
        ):
            raise ValueError(
                "Restart verification requires real uninterrupted updates one and two"
            )
        for index in range(number, until):
            tick = monotonic()
            guard(force=True)
            mx.random.seed(spec["seed"] + index)
            r = data[index % len(data)]
            parent_state = folder / f"step-{index:04d}" / "state.json"
            journal = folder / f"prepared-{index + 1:04d}"
            descriptor = journal / "update.json"
            replayed_derivative = descriptor.exists()
            if replayed_derivative:
                prepared = read(descriptor)
                if (
                    prepared["parent"] != digest(parent_state)
                    or prepared["row"] != r["id"]
                    or prepared["seed"] != spec["seed"] + index
                    or prepared["weights"] != digest(journal / "gradient.safetensors")
                ):
                    raise ValueError(
                        "Prepared update no longer matches its parent or row"
                    )
                grad = tree_unflatten(
                    list(mx.load(str(journal / "gradient.safetensors")).items())
                )
                values, magnitude = prepared["loss"], prepared["norm"]
            else:
                if restart_check:
                    raise ValueError(
                        "Exact optimizer replay requires the realized native derivative"
                    )
                grad, values, magnitude = derivative(m, step, r, spec, guard)
                if parent_state.exists():
                    with TemporaryDirectory(dir=folder) as tmp:
                        fresh = Path(tmp) / "update"
                        fresh.mkdir()
                        tensors(fresh / "gradient.safetensors", grad)
                        save(
                            fresh / "update.json",
                            {
                                "parent": digest(parent_state),
                                "row": r["id"],
                                "seed": spec["seed"] + index,
                                "weights": digest(fresh / "gradient.safetensors"),
                                "norm": magnitude,
                                "loss": values,
                            },
                        )
                        fresh.rename(journal)
            if pause_prepared and index + 1 == until:
                if not descriptor.exists():
                    raise ValueError("Pause requires a saved optimizer parent")
                save(
                    REPORT / "interrupted-before-update.json",
                    {
                        "next_update": until,
                        "prepared": digest(descriptor),
                        "parent": digest(parent_state),
                    },
                )
                return
            opt.update(step, grad)
            mx.eval(step.parameters(), opt.state)
            number = index + 1
            history.append(
                {
                    "step": number,
                    "row": r["id"],
                    "replayed_derivative": replayed_derivative,
                    "prepared": digest(descriptor) if descriptor.exists() else None,
                    "positions": len(r["p"]) + len(r["y"]) - 1,
                    "loss": values,
                    "gradient": magnitude,
                    "seconds": monotonic() - tick,
                    "peak_mlx": mx.get_peak_memory(),
                }
            )
            if restart_check:
                ref = folder / "step-0002"
                old = validate(ref, ident)["log"][-1]
                verified = {}
                for name, tree in (
                    ("controller", step.parameters()),
                    ("optimizer", opt.state),
                ):
                    prior = mx.load(str(ref / (name + ".safetensors")))
                    new = dict(tree_flatten(tree))
                    verified[name] = new.keys() == prior.keys() and all(
                        mx.array_equal(v, prior[k]).item() for k, v in new.items()
                    )
                verified["loss_gradient"] = (
                    values == old["loss"] and magnitude == old["gradient"]
                )
                save(
                    REPORT / "restart.json",
                    {
                        "passed": all(verified.values()),
                        "checks": verified,
                        "first": digest(folder / "step-0001/state.json"),
                        "boundary": "after differentiation, before optimizer application",
                        "realized_gradient": digest(journal / "gradient.safetensors"),
                        "second": digest(ref / "state.json"),
                    },
                )
                if not all(verified.values()):
                    raise ValueError("Restored next update differs")
                return
            if number in (1, 2, until) or number % 12 == 0:
                persist(folder, step, opt, number, ident, history)
        if number == spec["early_step"]:
            path = folder / f"step-{number:04d}"
            state = validate(path, ident)
            save(
                REPORT / "early-fit.json",
                {
                    "weights": state["hashes"]["controller.safetensors"],
                    "state": digest(path / "state.json"),
                    "identity": ident,
                    "updates": number,
                    "log": history,
                },
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--until", type=int, required=True)
    p.add_argument("--restart-check", action="store_true")
    p.add_argument("--pause-prepared", action="store_true")
    a = p.parse_args()
    fit(a.until, a.restart_check, a.pause_prepared)
