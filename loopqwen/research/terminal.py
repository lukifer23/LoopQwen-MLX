"""Compare data-matched dense and recurrent generation of the same requested final state."""

import gc
from pathlib import Path
from statistics import median
import mlx.core as mx
import numpy as np
from . import bootstrap, core
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save, checkpoint_location
from ..training.transfer7 import prompt


def question(row):
    return (
        prompt(row, "document").rsplit("\n", 1)[0]
        + " Follow exactly three references. Give only the final document number.\nFinal document:"
    )


def run():
    s = read("research/experiments/functional-terminal-01.json")
    name = s["id"]
    out = core.OUT / name
    if (out / "decision.json").exists():
        return
    b = core.Budget(name, s["limits"])
    try:
        groups = core.reserve(
            name,
            s["seed"],
            dict(train=s["train_maps"], dev=s["dev_maps"]),
            functional=True,
        )
        par = Path(s["parent_path"])
        if digest(par / "adapters.safetensors") != s["parent_sha256"]:
            raise ValueError("Parent adapter changed")
        result = {}
        for k in s["depths"]:
            arm = f"K{k}"
            folder = out / arm
            m, t = core.model(par, k)
            if k == s["depths"][0]:
                parent_retention = bootstrap.categorical(m, t, b)
            docs = [
                core.doc(t, question(r), " " + str(r["states"][2]))
                for r in groups["train"]
            ]
            cp = Path("checkpoints/research") / name / arm
            st = core.train(m, docs, dict(s, depth=k), cp, b, s["parent_sha256"])
            gen = checkpoint_location(cp)
            w = digest(gen / "adapters.safetensors")
            if not (folder / "model").exists():
                bootstrap.release(gen, folder / "model")
            records = []
            for r in groups["dev"]:
                ids = t.encode(question(r), add_special_tokens=False)
                v = core.generate(m, t, ids, s["generation_cap"], b.check, line=True)
                correct = v["complete"] and v["visible"].strip() == str(r["states"][2])
                records.append(dict(**r, **v, correct=correct))
            packed(folder / "eval.json.gz", dict(weights=w, records=records))
            result[arm] = dict(
                correct=sum(r["correct"] for r in records),
                completed=sum(r["complete"] for r in records),
                n=len(records),
                tokens=sum(len(r["token_ids"]) for r in records),
                median_ms=1000 * median(r["seconds"] for r in records),
                generation_seconds=sum(r["seconds"] for r in records),
                update_seconds=sum(r["seconds"] for r in st["history"]),
                train_targets=sum(r["targets"] for r in st["history"]),
                categorical=bootstrap.categorical(m, t, b),
                retention={str(j): bootstrap.retention(m, j, b) for j in (1, 3)},
            )
            del m, t
            gc.collect()
            mx.clear_cache()
        dense, loop = (
            unpack(out / f"K{j}" / "eval.json.gz")["records"] for j in s["depths"]
        )
        if [r["prompt_ids"] for r in dense] != [r["prompt_ids"] for r in loop]:
            raise ValueError("Unpaired terminal requests")
        delta = np.array(
            [
                int(r["correct"]) - int(d["correct"])
                for r, d in zip(loop, dense, strict=True)
            ]
        )
        idx = np.random.default_rng(s["seed"]).integers(
            len(loop), size=(5000, len(loop))
        )
        result.update(
            parent_categorical=parent_retention,
            paired=dict(
                wins=int((delta > 0).sum()),
                losses=int((delta < 0).sum()),
                ci=np.quantile(delta[idx].mean(1), [0.025, 0.975]).tolist(),
            ),
            passed=bool(
                delta.mean() >= 0.05
                and parent_retention[-1] - result["K3"]["categorical"][-1] <= 2
            ),
        )
        save(out / "decision.json", result)
    finally:
        b.close()


if __name__ == "__main__":
    run()
