"""Generate full-vocabulary judgments on counterfactual human-written premise groups."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import REVISION
from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..evaluation.relations import conclusion, summarize
from ..io import checkpoint_location, digest, read, save
from . import bootstrap, core, inference_data

NAME = "entailment-triplets-01"
REG = Path("research/experiments") / (NAME + ".json")
PANEL = Path("research/data") / (NAME + ".json")
TRAIN = PANEL.with_name("entailment-triplets-training.json")


def probe(m, t, ids):
    start = perf_counter()
    words = ("yes", "no", "maybe")
    letters = [t.encode(" " + v, add_special_tokens=False) for v in words]
    if any(len(v) != 1 for v in letters):
        raise ValueError("Relation probe must use one lexical token per class")
    # A shared prefix and unnormalized lexical logits avoid grading output formatting.
    x = ids + t.encode(
        "After comparing the two statements, my answer is", add_special_tokens=False
    )
    z = m.project(m.hidden(mx.array([x]))[:, -1])[:, [v[0] for v in letters]].astype(
        mx.float32
    )
    logits = z[0].tolist()
    return words[max(range(3), key=logits.__getitem__)], logits, perf_counter() - start


def evaluate(s, rows, model, weight, b, root):
    records = {"base": [], "candidate": []}
    retention = {}
    half = len(rows) // 2
    for offset, order in ((0, ("candidate", "base")), (half, ("base", "candidate"))):
        for arm in order:
            m, t, _ = base() if arm == "base" else load_trained(model)
            for row in rows[offset : offset + half]:
                ids = row["prompt_ids"]
                v = core.generate(
                    m, t, ids, s["generation_cap"], b.check, thinking=True
                )
                choice = conclusion(v["visible"]) if v["complete"] else None
                b.check()
                guess, logits, time = probe(m, t, ids)
                records[arm].append(
                    dict(
                        **v,
                        id=row["id"],
                        answer=choice,
                        correct=choice == row["label"],
                        probe=guess,
                        probe_logits=logits,
                        probe_seconds=time,
                    )
                )
                packed(
                    root / (arm + ".json.gz"),
                    {
                        "weights": REVISION if arm == "base" else weight,
                        "records": records[arm],
                    },
                )
            if offset and s.get("measure_retention", True):
                retention[arm] = bootstrap.retention(m, 1 if arm == "base" else 2, b)
            del m, t
            gc.collect()
            mx.clear_cache()
    return summarize(rows, records), retention


def passed(s, score, ret):
    a = score["all"]
    return (
        a["wins"] - a["losses"] >= s["net_floor"]
        and a["candidate"]["seconds"] <= a["base"]["seconds"]
        and a["candidate"]["triplets"] > a["base"]["triplets"]
        and a["candidate"]["probe"] > a["base"]["probe"]
        and all(score[k]["wins"] > score[k]["losses"] for k in ("esnli", "multinli"))
        and (not ret or ret["candidate"]["nll"] <= ret["base"]["nll"] + 0.03)
    )


def run(freeze):
    s = read(REG)
    panel = inference_data.material(s)
    docs, manifest = inference_data.training(s, panel)
    for p, v in ((PANEL, panel), (TRAIN, manifest)):
        if freeze and not p.exists():
            save(p, v)
        if read(p) != v:
            raise ValueError("Entailment material changed")
    if freeze:
        return
    registry = read("research/registry.json")
    if registry["experiments"][NAME]["registration"] != digest(REG) or any(
        registry["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ):
        raise ValueError("Unregistered entailment comparison")
    if any(
        digest(Path(s[k]) / "adapters.safetensors") != s[h]
        for k, h in (("parent", "parent_sha256"), ("dense_control", "dense_sha256"))
    ):
        raise ValueError("Wrong immutable model lineage")
    root = core.OUT / NAME
    root.mkdir(exist_ok=True, parents=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    b = core.Budget(NAME, s["limits"])
    res = {
        "registration": digest(REG),
        "training": digest(TRAIN),
        "data": digest(PANEL),
        "passed": False,
    }
    try:
        m, t, st = load_trained(s["parent"])
        m.checkpointing = True
        cp = Path("checkpoints/research") / NAME
        state = core.train(
            m,
            docs,
            dict(s, deployment=st["spec"]["deployment"]),
            cp,
            b,
            s["parent_sha256"],
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        weight = digest(gen / "adapters.safetensors")
        res.update(
            updates=state["step"],
            targets=sum(h["targets"] for h in state["history"]),
            weights=weight,
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        scores, ret = evaluate(s, panel, root / "model", weight, b, root)
        res.update(
            status="completed",
            scores=scores,
            retention=ret,
            passed=passed(s, scores, ret),
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        res.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", res)
    print(res["status"], res["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
