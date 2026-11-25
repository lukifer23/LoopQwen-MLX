"""Compare correctness-ranked sampled solutions with a frozen reference on native MLX."""

import argparse
import gc
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_map
from mlx_lm.utils import load_tokenizer

from .. import checkpoint
from ..data.tokenize import batch
from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import prompt_ids
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from ..training.losses import loss_sum
from . import bootstrap, checked_rationale, core, rationale_transfer

REG = Path("research/experiments/paired-solutions-01.json")
PANEL = Path("research/data/paired-solutions-01.json")
POOL = PANEL.with_name("paired-solutions-pool.json")


def material(s, exclude=()):
    old = read(rationale_transfer.DATA) + read(checked_rationale.PANEL)
    used = {" ".join(r["question"].split()) for r in old}
    used.update(" ".join(r["question"].split()) for r in exclude)
    reserve = read("research/data/gsm-equations-01.json")
    used.update(
        " ".join(r["question"].split())
        for r in reserve["train"]
        + reserve["dev"]
        + read("research/data/external-anchor.json")["gsm"]
    )
    checked = {r["question_id"] for r in read(checked_rationale.TRAIN)["selected"]}
    historical = set().union(
        *json.loads(
            gzip.decompress(
                Path("research/registry/historical-text-ids.json.gz").read_bytes()
            )
        ).values()
    )
    svamp = rationale_transfer.material(
        dict(s, sources={"svamp": s["sources"]["svamp"]}), used
    )
    used.update(r["question"] for r in svamp)
    path = Path("data/research_external/gsm-train.jsonl")
    if digest(path) != s["sources"]["gsm8k_reserved"]["sha256"]:
        raise ValueError("Wrong GSM8K publisher source")
    publisher = [json.loads(x) for x in path.read_bytes().splitlines()]
    t = load_tokenizer(local_snapshot())
    rows = []
    for i in sorted(range(len(publisher)), key=lambda i: fingerprint([s["seed"], i])):
        q = " ".join(publisher[i]["question"].split())
        if (
            q in used
            or fingerprint(q) in checked
            or any(
                hashlib.sha256(("text:" + q + suffix).encode()).hexdigest()
                in historical
                for suffix in (
                    "",
                    " Answer:",
                    " Show your steps. End with Final answer: <number>. Solution:",
                )
            )
        ):
            continue
        gold = numeric(publisher[i]["answer"].rsplit("####", 1)[1])
        if gold is None:
            continue
        ids = prompt_ids(t, {"question": q}, {"demos": 0})
        if len(ids) + s["rollout_cap"] > 384:
            continue
        used.add(q)
        rows.append(
            {
                "question": q,
                "id": fingerprint(q),
                "gold": gold,
                "prompt_ids": ids,
                "source": "gsm8k_reserved",
                "source_id": i,
            }
        )
        if len(rows) == s["pool"] + s["per_source"]:
            break
    if len(rows) != s["pool"] + s["per_source"]:
        raise ValueError("Fresh question pool exhausted")
    panel = [
        r for pair in zip(rows[: s["per_source"]], svamp, strict=True) for r in pair
    ]
    return rows[s["per_source"] :], panel


def preference(records):
    pairs = []
    for i, good in enumerate(records):
        if not good["correct"]:
            continue
        for j, bad in enumerate(records):
            if bad["answer"] is None or bad["correct"]:
                continue
            a, b = len(good["token_ids"]), len(bad["token_ids"])
            if max(a, b) <= 1.5 * min(a, b):
                pairs.append((abs(a - b), i, j))
    return min(pairs)[1:] if pairs else None


def sample(m, t, items, s, budget, root):
    draws, chosen = [], []
    for i, row in enumerate(items):
        records = []
        draws.append({"id": row["id"], "attempts": records})
        for j in range(s["draws"]):
            v = core.generate(
                m,
                t,
                row["prompt_ids"],
                s["rollout_cap"],
                budget.check,
                thinking=True,
                temperature=s["temperature"],
                seed=s["seed"] + i * s["draws"] + j,
            )
            answer = numeric(v["visible"]) if v["complete"] else None
            records.append(dict(**v, answer=answer, correct=answer == row["gold"]))
            pair = preference(records)
            if pair is not None:
                chosen.append({"pool_index": i, "chosen": pair[0], "rejected": pair[1]})
                break
        packed(root / "rollouts.json.gz", draws)
        save(root / "pairs.json", chosen)
        if len(chosen) == s["max_pairs"]:
            break
    return draws, chosen


def document(v):
    return {"ids": v["prompt_ids"] + v["token_ids"], "prefix": len(v["prompt_ids"])}


def preference_terms(chosen_nll, rejected_nll, reference, beta):
    margin = beta * ((reference[0] - chosen_nll) - (reference[1] - rejected_nll))
    loss = max(0.0, -margin) + math.log1p(math.exp(-abs(margin)))
    return loss, beta * mx.sigmoid(mx.array(-margin)).item()


def train_preferences(m, s, pairs, draws, budget, root, cp, deployment):
    size = s["logical_batch"]
    count = len(pairs) // size * size
    docs = []
    costs = []
    for p in pairs[:count]:
        a = draws[p["pool_index"]]["attempts"]
        pair = [document(a[p[k]]) for k in ("chosen", "rejected")]
        docs.append(pair)
        budget.check()
        costs.append([loss_sum(m, *batch([v])).item() for v in pair])
    save(root / "reference-nll.json", costs)
    history = []
    state = {
        "identity": fingerprint(
            [
                digest(REG),
                s["parent_sha256"],
                digest(root / "rollouts.json.gz"),
                pairs,
                costs,
            ]
        ),
        "parent": s["parent_sha256"],
        "spec": dict(s, deployment=deployment),
        "step": 0,
        "history": history,
    }
    cp.mkdir(parents=True, exist_ok=False)
    m.checkpointing = True
    m.train()
    opt = optim.AdamW(learning_rate=s["lr"], weight_decay=0, bias_correction=True)
    vg = nn.value_and_grad(m, loss_sum)
    try:
        for epoch in range(s["epochs"]):
            order = list(range(count))
            random.Random(s["seed"] + epoch).shuffle(order)
            for k in range(0, count, size):
                start = perf_counter()
                grads = None
                losses = []
                for idx in order[k : k + size]:
                    budget.check()
                    results = []
                    for doc in docs[idx]:
                        loss, gradient = vg(m, *batch([doc]))
                        mx.eval(loss, gradient)
                        results.append((loss.item(), gradient))
                    (a, ga), (b, gb) = results
                    loss, slope = preference_terms(a, b, costs[idx], s["beta"])
                    losses.append(loss)
                    # Chain rule for sequence-summed DPO. Each backward pass holds one response's activations.
                    factor = slope / size
                    delta = tree_map(lambda x, y, scale=factor: scale * (x - y), ga, gb)
                    grads = (
                        delta
                        if grads is None
                        else tree_map(lambda x, y: x + y, grads, delta)
                    )
                    mx.eval(grads)
                grads, norm = optim.clip_grad_norm(grads, s["clip"])
                if not math.isfinite(norm.item() + sum(losses)):
                    raise FloatingPointError("Nonfinite preference objective")
                opt.update(m, grads)
                mx.eval(m.trainable_parameters(), opt.state)
                state["step"] += 1
                history.append(
                    {
                        "epoch": epoch,
                        "indices": order[k : k + size],
                        "loss": sum(losses) / size,
                        "seconds": perf_counter() - start,
                    }
                )
            checkpoint.write(cp, m, opt, state)
    finally:
        latest = cp / "latest.json"
        if state["step"] and (
            not latest.exists()
            or read(checkpoint_location(cp) / "state.json")["step"] != state["step"]
        ):
            checkpoint.write(cp, m, opt, state)
    return state


def run(freeze):
    s = read(REG)
    pool, panel = material(s)
    for path, value in ((POOL, pool), (PANEL, panel)):
        if freeze and not path.exists():
            save(path, value)
        if read(path) != value or (
            not freeze
            and digest(path)
            != read("research/registry.json")["datasets"][str(path)]["sha256"]
        ):
            raise ValueError("Frozen preference cohort changed")
    if freeze:
        return
    root = core.OUT / s["id"]
    if root.exists():
        raise FileExistsError(root)
    for key, sha in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(s[key]) / "adapters.safetensors") != s[sha]:
            raise ValueError("Wrong paired parents")
    core.assets()
    budget = core.Budget(s["id"], s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "data": digest(PANEL),
        "pool": digest(POOL),
    }
    cp = Path("checkpoints/research") / s["id"]
    try:
        m, t, parent = load_trained(s["parent"])
        draws, pairs = sample(m, t, pool, s, budget, root)
        result.update(
            eligible=len(pairs),
            sampled_questions=len(draws),
            sampling_seconds=sum(v["seconds"] for r in draws for v in r["attempts"]),
        )
        if len(pairs) < s["min_pairs"]:
            result.update(
                status="insufficient_pairs",
                reason="Too few closely length-matched correct/wrong response pairs",
            )
        else:
            st = train_preferences(
                m, s, pairs, draws, budget, root, cp, parent["spec"]["deployment"]
            )
            gen = checkpoint_location(cp)
            bootstrap.release(gen, root / "model")
            result.update(
                weights=digest(gen / "adapters.safetensors"),
                updates=st["step"],
                update_seconds=sum(h["seconds"] for h in st["history"]),
            )
            del m, t
            gc.collect()
            mx.clear_cache()
            rationale_transfer.evaluate_pair(
                root,
                root / "model",
                panel,
                s,
                result["weights"],
                budget,
                result,
                retention=True,
            )
            full, short, r = (result[k] for k in ("full", "prefix_256", "retention"))
            a = full["all"]
            result["passed"] = (
                a["wins"] - a["losses"] >= 2
                and a["p"] < 0.05
                and a["candidate"]["seconds"] <= a["base"]["seconds"]
                and all(
                    full[src]["wins"] >= full[src]["losses"]
                    and short[src]["wins"] > short[src]["losses"]
                    for src in s["sources"]
                )
                and r["candidate"]["nll"] <= r["base"]["nll"] + 0.03
            )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        budget.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
