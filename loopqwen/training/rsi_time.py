"""Spend the measured adaptive diagnosis overhead on ordinary, matched static updates."""

from pathlib import Path
from time import perf_counter
import signal
import subprocess
import mlx.core as mx
import mlx.optimizers as optim
import psutil
from .. import checkpoint
from ..data import pointer6
from ..data.curriculum import REG, S, pools
from ..identity import fingerprint
from ..io import read, save, digest, checkpoint_location
from ..evaluation.rsi import score
from .exposure import ARM, verify_optimizer
from .mechanism import model_for, requests, assets
from . import pointer5 as kernel


def run():
    src = Path("docs/measurements/rsi1")
    orig = read(src / "decision.json")
    if not orig["promoted"]:
        raise ValueError("Ineligible comparison")
    root = Path("outputs/rsi_time1")
    cp = Path("checkpoints/rsi_time1")
    if (root / "decision.json").exists():
        return print(read(root / "decision.json")["status"])
    cp.mkdir(parents=True, exist_ok=False)
    spec = read("experiments/rsi_followup.json")
    wall = perf_counter()
    budget = sum(
        sum(v.values()) * (1 if k.startswith("adaptive") else -1)
        for k, v in orig["timers"].items()
    )
    source = read(src / "static2-lineage.json")
    gen = Path(source["generation"])
    if (
        digest(gen / "optimizer.safetensors") != source["optimizer"]
        or digest(gen / "adapters.safetensors") != source["weights"]
    ):
        raise ValueError("Static parent mismatch")
    assets()
    m, t, _ = model_for(ARM)
    opt = optim.AdamW(learning_rate=S["lr"], weight_decay=0, bias_correction=True)
    checkpoint.restore(gen, m, opt)
    verify_optimizer(m, opt, 32, S["lr"])
    splits, _ = pools()
    tick = perf_counter()
    spare = []
    used = []
    for j in (1, 2):
        chosen = {x["id"] for x in read(src / f"static{j}-curriculum.json")["selected"]}
        spare.extend(r for r in splits[f"pool{j}"] if r["map_id"] not in chosen)
        used.extend(r for r in splits[f"pool{j}"] if r["map_id"] in chosen)

    def balanced(rows):
        bins = {}
        for r in sorted(rows, key=lambda r: fingerprint(["time-control", r["map_id"]])):
            bins.setdefault(r["stratum"], []).append(r)
        return [
            r
            for batch in zip(*(bins[k] for k in sorted(bins)), strict=True)
            for r in batch
        ]

    order = balanced(spare) + balanced(used)
    select_seconds = perf_counter() - tick
    history = []
    spent = select_seconds
    status = "time matched"
    interrupt = []

    def guard():
        if interrupt:
            raise InterruptedError("Interrupted")
        if perf_counter() - wall > spec["static_max_wall_seconds"] - 3:
            raise TimeoutError("Wall cap")
        if (
            mx.get_peak_memory() > spec["peak_gib"] * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            in ("2", "4")
        ):
            raise MemoryError("Resource stop")

    handlers = {
        s: signal.signal(s, lambda n, f: interrupt.append(n))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    fn = kernel.make_update(m, opt, "recurrent")
    m.train()
    mx.reset_peak_memory()
    try:
        for i in range(spec["static_max_extra_updates"]):
            guard()
            forecast = 1.3 * max([1.0] + [r["seconds"] for r in history[-6:]])
            if spent + forecast >= budget:
                break
            depth = 4 + (32 + i) % 3
            group = [
                pointer6.ordered(r, "randomized", S["seed"], 3)
                for r in order[i * 8 : i * 8 + 8]
            ]
            r = fn(requests(group, depth, ARM, t, {}), depth, S["position_cap"])
            r["ids"] = [r["map_id"] for r in group]
            history.append(r)
            spent += r["seconds"]
    except (MemoryError, TimeoutError, InterruptedError, FloatingPointError) as err:
        status = str(err)
    finally:
        st = dict(
            step=int(opt.step.item()),
            parent=source,
            status=status,
            history=history,
            policy=spec,
        )
        checkpoint.write(cp, m, opt, st)
        for s, h in handlers.items():
            signal.signal(s, h)
    wh = digest(checkpoint_location(cp) / "adapters.safetensors")
    evaluation = None
    if status == "time matched":
        _, evaluation = score(
            m, t, ARM, splits["development"], root / "evaluation.json.gz", wh, guard
        )
    save(
        root / "decision.json",
        dict(
            status=status,
            weights=wh,
            optimizer=digest(checkpoint_location(cp) / "optimizer.safetensors"),
            parent=source["weights"],
            generation=str(checkpoint_location(cp)),
            extra_budget=budget,
            extra_spent=spent,
            selection=select_seconds,
            history=history,
            evaluation=evaluation,
            seconds=perf_counter() - wall,
        ),
    )
    print(status, len(history), spent, flush=True)


if __name__ == "__main__":
    run()
