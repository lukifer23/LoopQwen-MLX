"""Disentangle restoration, physical array layout and native update repeatability."""

import argparse
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from loopqwen.middle_depth import architecture, donor
from loopqwen.training.optimizers import optimizer
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_fit import RUN, SPEC, loss, restore, update
from scripts.middle_depth_material import ROWS


def audit(arm):
    spec = read(SPEC)
    folder = RUN / str(spec["seed"]) / arm
    state = read(folder / "step-0001/state.json")
    rows = unpack(ROWS)[2:4]
    with worker("profile", "restart diagnosis/" + arm) as guard:
        text, _ = donor(BASE, "base")
        model = architecture(text, state["identity"]["depth"], spec["seed"])
        model.train()
        opt = optimizer(
            dict(
                learning_rate=spec["lr"],
                weight_decay=0.01,
                warmup=spec["warmup"],
                steps=spec["steps"],
            )
        )
        grad = nn.value_and_grad(model, loss)

        def params():
            return dict(
                tree_flatten(
                    {"model": model.trainable_parameters(), "optimizer": opt.state}
                )
            )

        def differences(a, b):
            arrays = {}
            for k, v in a.items():
                value = v.astype(mx.float32)
                ref = b[k].astype(mx.float32)
                diff = mx.linalg.norm(value - ref).item()
                arrays[k] = dict(
                    max=mx.abs(value - ref).max().item(),
                    l2=diff,
                    relative=diff / max(1e-20, mx.linalg.norm(ref).item()),
                )
            return arrays

        step1 = None
        candidates = []
        history = []
        expected = mx.load(str(ROOT / f"restart-{arm}.safetensors"))
        for _ in range(2):
            restore(folder, model, opt, state["identity"])
            start = params()
            if step1 is None:
                step1 = start
            else:
                assert all(mx.array_equal(v, step1[k]).item() for k, v in start.items())
            rec = update(
                model,
                opt,
                grad,
                rows,
                1,
                state["identity"]["depth"],
                state["identity"]["feature_id"],
                spec,
                guard,
            )
            history.append(rec)
            candidates.append(params())
        save(
            ROOT / f"restart-{arm}-audit.json",
            dict(
                original=differences(candidates[0], expected),
                repeated=differences(candidates[1], candidates[0]),
                history=history,
                reference_delta=differences(expected, step1),
            ),
        )
        print("scores", history[0]["exit_loss"], history[1]["exit_loss"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm")
    audit(p.parse_args().arm)
