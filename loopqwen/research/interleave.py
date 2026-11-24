"""Hold fresh maps and objective fixed while replacing two replay slots per update."""

import gc
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from . import bootstrap, core, replay
from .. import checkpoint
from ..data import pointer5, pointer6
from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import read, save, checkpoint_location, digest
from ..training.exposure import ARM, verify_optimizer
from ..training.mechanism import requests

NAME = "functional-interleave-01"
SCHEDULE = Path("research/data") / (NAME + ".json")


def data(s):
    src = Path("research/data") / (s["dataset"] + ".json.gz")
    if digest(src) != read("research/registry.json")["datasets"][str(src)]["sha256"]:
        raise ValueError("Functional pool differs")
    groups = unpack(src)["splits"]
    pool = groups["pool1"]
    pointer5.build()
    old = pointer5.rows("train")
    n = s["train_updates"] * (s["batch"] - s["functional_per_update"])
    shuffled = lambda rows: sorted(
        rows, key=lambda r: fingerprint([s["seed"], r["map_id"]])
    )[:n]
    old = [pointer6.ordered(r, "random0", s["seed"]) for r in shuffled(old)]
    material = dict(
        fresh=pool,
        control=shuffled(pool),
        permutation=old,
        dev_ids=[r["map_id"] for r in groups["dev"]],
    )
    if SCHEDULE.exists() and read(SCHEDULE) != material:
        raise ValueError("Replay schedule changed")
    if not SCHEDULE.exists():
        save(SCHEDULE, material)
    return material, groups["dev"]


def train(s, m, t, schedule, arm, b):
    cp = Path("checkpoints/research") / NAME / arm
    cp.mkdir(parents=True, exist_ok=True)
    key = fingerprint(
        [
            s,
            arm,
            core.assets(),
            schedule,
            digest(Path(__file__)),
            digest(Path(replay.__file__)),
        ]
    )
    opt = optim.AdamW(learning_rate=s["lr"], weight_decay=0, bias_correction=True)
    st = dict(identity=key, step=0, parent=s["parent_sha256"], history=[])
    if (cp / "latest.json").exists():
        st = checkpoint.restore(cp, m, opt)
        if st["identity"] != key:
            raise ValueError("Incompatible interleave recovery")
        verify_optimizer(m, opt, st["step"], s["lr"])
    fn = replay.updates(m, opt, s)
    fresh = s["functional_per_update"]
    extra = s["batch"] - fresh
    docs = requests(schedule["fresh"], s["depth"], ARM, t, {})
    other = requests(schedule[arm], s["depth"], ARM, t, {})
    m.train()
    saved = st["step"]
    try:
        for i in range(saved, s["train_updates"]):
            b.check()
            batch = (
                docs[i * fresh : (i + 1) * fresh] + other[i * extra : (i + 1) * extra]
            )
            if len(batch) != s["batch"]:
                raise ValueError("Incomplete logical batch")
            st["history"].append(fn(batch))
            st["step"] += 1
            if st["step"] % 8 == 0:
                checkpoint.write(cp, m, opt, st)
                saved = st["step"]
    finally:
        if saved != st["step"]:
            checkpoint.write(cp, m, opt, st)
    return checkpoint_location(cp), st


def run():
    s = read(Path("research/experiments") / (NAME + ".json"))
    out = core.OUT / NAME
    if core.archived_result(out, s):
        return
    b = core.Budget(NAME, s["limits"])
    try:
        schedule, dev = data(s)
        par = Path(s["parent_path"])
        if digest(par / "adapters.safetensors") != s["parent_sha256"]:
            raise ValueError("Parent differs")
        base = read("research/results/functional-replay-cap84/decision.json")["parent"]
        result = dict(parent=base)
        for arm in ("control", "permutation"):
            m, t = core.model(par, s["depth"])
            gen, st = train(s, m, t, schedule, arm, b)
            folder = out / arm
            if not (folder / "model").exists():
                bootstrap.release(gen, folder / "model")
            w = digest(gen / "adapters.safetensors")
            result[arm] = dict(
                **replay.evaluate(m, t, dev, s, b, folder / "eval.json.gz", w),
                weights=w,
                update_seconds=sum(r["seconds"] for r in st["history"]),
            )
            del m, t
            gc.collect()
            mx.clear_cache()

        def hits(path):
            rows = unpack(path)["records"]
            if [r["map_id"] for r in rows] != schedule["dev_ids"]:
                raise ValueError("Wrong cohort")
            return np.array([int(r["predicted"][-1] == r["states"][-1]) for r in rows])

        treatment = hits(out / "permutation/eval.json.gz")
        controls = {
            "parent": Path("research/results/functional-replay-cap84/initial.json.gz"),
            "control": out / "control/eval.json.gz",
        }
        result["paired"] = {}
        idx = np.random.default_rng(s["seed"]).integers(len(dev), size=(5000, len(dev)))
        for k, path in controls.items():
            d = treatment - hits(path)
            result["paired"][k] = dict(
                wins=int((d > 0).sum()),
                losses=int((d < 0).sum()),
                ci=np.quantile(d[idx].mean(1), [0.025, 0.975]).tolist(),
            )
        gain = treatment.mean() - base["functional"]["correct"][-1] / len(dev)
        result["passed"] = bool(
            gain >= 0.10
            and base["anchor"][-1] - result["permutation"]["anchor"][-1] <= 2
        )
        save(out / "decision.json", result)
    finally:
        b.close()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["data"]:
        data(read(Path("research/experiments") / (NAME + ".json")))
    else:
        run()
