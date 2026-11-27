"""A durable pre-update boundary for Metal VJPs with non-bitwise repeatability."""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import architecture, donor
from loopqwen.training.optimizers import optimizer
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_fit import (
    RUN,
    SPEC,
    checkpoint,
    loss,
    restore,
    tensors,
    update,
)
from scripts.middle_depth_material import ROWS


def boundary(arm, action):
    cfg = read(SPEC)
    folder = RUN / str(cfg["seed"]) / arm
    latest = read(folder / "latest.json")
    identity = read(folder / latest["path"] / "state.json")["identity"]
    stage = folder / ("prepared-" + latest["path"])
    stage.mkdir(exist_ok=True)
    meta = stage / "update.json"
    with worker("fit", f"{arm}/{action} prepared update") as guard:
        text, _ = donor(BASE, identity["donor"])
        m = architecture(text, identity["depth"], cfg["seed"])
        m.train()
        opt = optimizer(
            dict(
                learning_rate=cfg["lr"],
                weight_decay=0.01,
                warmup=cfg["warmup"],
                steps=cfg["steps"],
            )
        )
        step, log = restore(folder, m, opt, identity)
        deck = unpack(ROWS)
        rows = [
            deck[(step * cfg["batch"] + j) % len(deck)] for j in range(cfg["batch"])
        ]
        weights = stage / "gradient.safetensors"
        reference = stage / "next.safetensors"
        if action == "prepare":
            if meta.exists():
                raise FileExistsError(meta)

            class Capture:
                state = opt.state

                def update(self, model, g):
                    tensors(weights, g)
                    opt.update(model, g)
                    self.state = opt.state

            rec = update(
                m,
                Capture(),
                nn.value_and_grad(m, loss),
                rows,
                step,
                identity["depth"],
                identity["feature_id"],
                cfg,
                guard,
            )
            tensors(
                reference, {"model": m.trainable_parameters(), "optimizer": opt.state}
            )
            save(
                meta,
                dict(
                    start=latest["state"],
                    gradient=digest(weights),
                    reference=digest(reference),
                    log=rec,
                    step=step,
                    order=identity["order"],
                    next_rng_seed=cfg["seed"] + step,
                ),
            )
        else:
            data = read(meta)
            assert (
                data["start"] == latest["state"]
                and data["step"] == step
                and data["gradient"] == digest(weights)
                and data["reference"] == digest(reference)
                and data["order"] == identity["order"]
                and data["next_rng_seed"] == cfg["seed"] + step
                and data["log"]["ids"] == [r["id"] for r in rows]
            )
            mx.random.seed(cfg["seed"] + step)
            g = tree_unflatten(list(mx.load(str(weights)).items()))
            guard(force=True)
            opt.update(m, g)
            mx.eval(m.trainable_parameters(), opt.state)
            values = dict(
                tree_flatten(
                    {"model": m.trainable_parameters(), "optimizer": opt.state}
                )
            )
            expected = mx.load(str(reference))
            assert values.keys() == expected.keys()
            checks = {
                k: mx.array_equal(v, expected[k]).item() for k, v in values.items()
            }
            save(
                ROOT / f"prepared-restart-{arm}.json",
                dict(
                    passed=all(checks.values()),
                    checks=checks,
                    boundary=digest(meta),
                    gradient=digest(weights),
                    start=latest["state"],
                    step=step + 1,
                ),
            )
            if not all(checks.values()):
                raise ValueError("Prepared update failed exact cold replay")
            checkpoint(
                folder,
                m,
                opt,
                step + 1,
                identity,
                log + [data["log"]],
                cfg["seed"],
                identity["depth"],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("loop", "ordinary"))
    p.add_argument("action", choices=("prepare", "commit"))
    a = p.parse_args()
    boundary(a.arm, a.action)
