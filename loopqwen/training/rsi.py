"""Two fixed rounds with opt-in model-directed selection and immutable generations."""

from collections import Counter
from pathlib import Path
from time import perf_counter
import signal
import subprocess
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import psutil
from .. import checkpoint
from ..data import pointer6
from ..data.curriculum import REG, S, pools, select
from ..evaluation.mechanism import packed, unpack
from ..evaluation.rsi import score
from ..identity import fingerprint
from ..io import digest, read, save, checkpoint_location
from . import pointer5 as kernel
from .exposure import ARM, verify_optimizer
from .mechanism import model_for, requests, assets

ROOT = Path("outputs") / S["id"]
CHECK = Path("checkpoints") / S["id"]


def run():
    closed = ROOT / "decision.json"
    if closed.exists():
        return print(read(closed)["status"])
    if (ROOT / "start.json").exists() or CHECK.exists():
        raise ValueError("Incomplete prior experiment; preserve this lineage")
    if (
        S["id"] == "rsi2"
        and not read("docs/measurements/rsi1/decision.json")["promoted"]
    ):
        raise ValueError("Replication prerequisite failed")
    start = perf_counter()
    parent = Path(S["parent"])
    if digest(parent / "adapters.safetensors") != S["weights"]:
        raise ValueError("Parent digest mismatch")
    source = Path(__file__)
    ident = dict(
        registration=digest(REG),
        trainer=digest(source),
        selector=digest(Path("loopqwen/data/curriculum.py")),
        scorer=digest(Path("loopqwen/evaluation/rsi.py")),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        assets=assets(),
    )
    save(ROOT / "start.json", ident)
    initial = perf_counter()
    splits, excluded = pools()
    data_seconds = perf_counter() - initial
    packed(
        ROOT / "pools.json.gz", {k: v for k, v in splits.items() if k != "confirmation"}
    )
    save(
        ROOT / "data.json",
        dict(
            identities={k: fingerprint(v) for k, v in splits.items()},
            excluded_hash=excluded,
            generation_verification_seconds=data_seconds,
        ),
    )
    m, t, _ = model_for(ARM)
    canceled = []
    handlers = {
        s: signal.signal(s, lambda n, f: canceled.append(n))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    summary = {}
    timers = {}
    last = {}
    total_updates = 0

    def guard():
        if canceled:
            raise InterruptedError("Signal")
        if (
            perf_counter() - start
            >= min(S["max_gpu_seconds"], S["max_wall_seconds"]) - 3
        ):
            raise TimeoutError("Time cap")
        if (
            mx.get_peak_memory() > S["peak_gib"] * 2**30
            or psutil.virtual_memory().available < S["available_gib"] * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            in ("2", "4")
        ):
            raise MemoryError("Resource stop")

    def validate(name, w):
        _, s = score(
            m,
            t,
            ARM,
            splits["development"],
            ROOT / (name + "-evaluation.json.gz"),
            w,
            guard,
        )
        summary[name] = s
        save(ROOT / "scoreboard.json", summary)

    status = "completed"
    try:
        checkpoint.restore(parent, m)
        mx.reset_peak_memory()
        validate("parent", S["weights"])
        if summary["parent"]["accuracy"][-1] >= 0.95:
            raise ArithmeticError("Parent ceiling")
        for j in (1, 2):
            for branch in ("static", "adaptive") if j == 1 else ("adaptive", "static"):
                name = f"{branch}{j}"
                pool = splits[f"pool{j}"]
                root = CHECK / name
                old = last.get(branch, parent)
                wh = digest(old / "adapters.safetensors")
                opt = optim.AdamW(
                    learning_rate=S["lr"], weight_decay=0, bias_correction=True
                )
                checkpoint.restore(old, m, opt if j > 1 else None)
                if j > 1:
                    verify_optimizer(m, opt, S["updates_per_round"], S["lr"])
                diag = None
                budget = timers[name] = dict(candidate=0, selection=0, update=0)
                if branch == "adaptive":
                    diag, measure = score(
                        m, t, ARM, pool, ROOT / (name + "-diagnosis.json.gz"), wh, guard
                    )
                    budget["candidate"] = measure["seconds"]
                tick = perf_counter()
                chosen = select(pool, diag, j)
                curriculum = [
                    dict(id=r["map_id"], reason=why, stratum=r["stratum"])
                    for r, why in chosen
                ]
                save(
                    ROOT / (name + "-curriculum.json"),
                    dict(
                        parent=wh,
                        selected=curriculum,
                        counts=dict(Counter(r["reason"] for r in curriculum)),
                    ),
                )
                budget["selection"] = perf_counter() - tick
                root.mkdir(parents=True, exist_ok=False)
                fn = kernel.make_update(m, opt, "recurrent")
                m.train()
                mx.reset_peak_memory()
                st = dict(
                    parent=wh,
                    identity=ident,
                    round=j,
                    branch=branch,
                    step=(j - 1) * S["updates_per_round"],
                    history=[],
                    curriculum=fingerprint(curriculum),
                    status="interrupted",
                )
                try:
                    for i in range(S["updates_per_round"]):
                        guard()
                        depth = S["depths"][i % len(S["depths"])]
                        batch = [
                            pointer6.ordered(r, "randomized", S["seed"], j)
                            for r, _ in chosen[i * 8 : (i + 1) * 8]
                        ]
                        r = fn(
                            requests(batch, depth, ARM, t, {}), depth, S["position_cap"]
                        )
                        budget["update"] += r["seconds"]
                        st["history"].append(r)
                        st["step"] += 1
                        total_updates += 1
                        if int(opt.step.item()) != st["step"]:
                            raise ValueError("Cursor mismatch")
                    guard()
                    st["status"] = "completed"
                finally:
                    checkpoint.write(root, m, opt, st)
                    gen = checkpoint_location(root)
                    last[branch] = gen
                    save(
                        ROOT / (name + "-lineage.json"),
                        dict(
                            parent=wh,
                            weights=digest(gen / "adapters.safetensors"),
                            optimizer=digest(gen / "optimizer.safetensors"),
                            generation=str(gen),
                            step=st["step"],
                            status=st["status"],
                            history=st["history"],
                        ),
                    )
                validate(name, digest(gen / "adapters.safetensors"))
            if (
                min(summary[f"{b}{j}"]["accuracy"][2] for b in ("static", "adaptive"))
                < summary["parent"]["accuracy"][2] - 0.02
            ):
                raise ArithmeticError("Third-hop retention floor")
    except (
        MemoryError,
        TimeoutError,
        InterruptedError,
        FloatingPointError,
        ArithmeticError,
    ) as e:
        status = str(e)
    except Exception:
        status = "failed integrity or execution"
        raise
    finally:
        for sig, h in handlers.items():
            signal.signal(sig, h)
        result = dict(
            status=status,
            updates=total_updates,
            timers=timers,
            seconds=perf_counter() - start,
            summary=summary,
        )
        if "static2" in summary and "adaptive2" in summary:

            def per_map(name):
                rows = unpack(ROOT / (name + "-evaluation.json.gz"))["records"]
                return {
                    r["map_id"]: np.mean(
                        [q["correct"][-1] for q in rows if q["map_id"] == r["map_id"]]
                    )
                    for r in rows
                }

            a, b = per_map("adaptive2"), per_map("static2")
            if a.keys() != b.keys():
                raise ValueError("Unpaired maps")
            diff = np.array([a[k] - b[k] for k in sorted(a)])
            ci = np.quantile(
                np.random.default_rng(S["seed"])
                .choice(diff, (10000, len(diff)))
                .mean(axis=1),
                [0.025, 0.975],
            )
            result.update(
                difference=diff.mean(),
                interval=ci.tolist(),
                promoted=bool(status == "completed" and diff.mean() >= 0.03),
            )
        save(closed, result)
        print(
            status,
            total_updates,
            [(k, round(v["accuracy"][-1], 4)) for k, v in summary.items()],
            flush=True,
        )


if __name__ == "__main__":
    run()
