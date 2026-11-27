"""Resume the registered AdamW trajectory with a qualified layerwise native VJP."""

import argparse
from functools import partial
from pathlib import Path

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import architecture
from loopqwen.ordinary_model import ordinary
from loopqwen.training.optimizers import optimizer
from scripts import middle_depth_adjoint
from scripts import middle_depth_fit as primary
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_material import ROWS
from scripts.middle_depth_recover import reconcile

QUALIFICATION = ROOT / "adjoint-recovery.json"


def qualify():
    code = digest(Path(middle_depth_adjoint.__file__))
    result = {}
    for arm in ("loop", "ordinary"):
        path = ROOT / f"adjoint-{arm}.json"
        proof = read(path)
        assert proof["passed"] and proof["implementation"] == code
        result[arm] = digest(path)
    receipt = {
        "implementation": code,
        "numerical_checks": result,
        "plan": digest(Path("research/diagnostics/middle-depth/adjoint-recovery.md")),
        "baseline_trainer": digest(Path(primary.__file__)),
        "scope": "Whole training targets, native VJP at each layer, tied gradients summed before clipping and one AdamW update.",
    }
    if QUALIFICATION.is_file() and read(QUALIFICATION) != receipt:
        raise ValueError(
            "Checked gradients or original trainer changed after qualification"
        )
    save(QUALIFICATION, receipt)
    return receipt


def gradient(guard):
    from scripts.middle_depth_adjoint import bounded

    qualify()
    return partial(bounded, guard=guard)


def fit(arm, until):
    kval = qualify()
    cfg = read(primary.SPEC)
    root = primary.RUN / str(cfg["seed"]) / arm
    deck = unpack(ROWS)
    assert (
        1 <= until <= cfg["steps"]
        and len(deck) * cfg["epochs"] == cfg["steps"] * cfg["batch"]
    )
    with worker("fit", f"layerwise/{arm}/{until}") as guard:
        material = reconcile(root)
        st = read(root / material["path"] / "state.json")
        ident = st["identity"]
        assert ident["spec"] == digest(primary.SPEC) and ident["data"] == digest(ROWS)
        assert ident["trainer"] == kval["baseline_trainer"] and ident[
            "architecture"
        ] == digest(Path("loopqwen/middle_depth.py"))
        assert ident["order"] == fingerprint(
            [r["id"] for _ in range(cfg["epochs"]) for r in deck]
        )
        depth = 3 if arm == "loop" else 1
        assert ident["depth"] == depth
        assert ident["donor"] == "base"
        native, tok, _ = ordinary(BASE, require_release=True, lazy=True)
        model = architecture(native.text, depth, cfg["seed"])
        del native, tok
        model.train()
        opt = optimizer(
            {
                "learning_rate": cfg["lr"],
                "weight_decay": 0.01,
                "warmup": cfg["warmup"],
                "steps": cfg["steps"],
            }
        )
        step, log = primary.restore(root, model, opt, ident)
        grad = gradient(guard)
        engine = digest(Path(__file__))
        schedule = digest(QUALIFICATION)
        saved = step
        try:
            while step < until:
                guard(force=True)
                rows = [
                    deck[(step * cfg["batch"] + i) % len(deck)]
                    for i in range(cfg["batch"])
                ]
                rec = primary.update(
                    model, opt, grad, rows, step, depth, ident["feature_id"], cfg, guard
                )
                rec.update(adjoint_qualification=schedule, continuation_trainer=engine)
                step += 1
                log.append(rec)
                if step % cfg["checkpoint_every"] == 0 or step == until:
                    primary.checkpoint(
                        root, model, opt, step, ident, log, cfg["seed"], depth
                    )
                    saved = step
                    print(
                        arm,
                        step,
                        round(rec["seconds"], 2),
                        "s",
                        round(mx.get_peak_memory() / 2**30, 2),
                        "GiB",
                        flush=True,
                    )
        except InterruptedError:
            if step > saved:
                primary.checkpoint(
                    root, model, opt, step, ident, log, cfg["seed"], depth
                )
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("loop", "ordinary"))
    p.add_argument("--until", type=int, required=True)
    a = p.parse_args()
    fit(a.arm, a.until)
