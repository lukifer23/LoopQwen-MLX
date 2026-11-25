"""Train on verified final operations and score fresh public first turns at two budgets."""

import argparse
from fractions import Fraction
import gc
import gzip
import hashlib
import json
import operator
from pathlib import Path
import re

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import prompt_ids
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import bootstrap, core, rationale_transfer

REG = Path("research/experiments/checked-rationale-01.json")
PANEL = Path("research/data/checked-rationale-01.json")
TRAIN = PANEL.with_name("checked-rationale-training.json")
DECIMAL = r"-?\d+(?:\.\d+)?"
EQUATION = re.compile(rf"({DECIMAL})([+*/-])({DECIMAL})=({DECIMAL})")
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
INVERSE = {"+": "-", "-": "+", "*": "/", "/": "*"}


def checked_response(solution):
    explanation, final = solution.rsplit("####", 1)
    gold = numeric(final)
    annotations = re.findall(r"<<([^<>]+)>>", explanation)
    if not annotations or gold is None:
        return None
    eq = EQUATION.fullmatch(annotations[-1].replace(" ", "").replace(",", ""))
    if eq is None:
        return None
    a, op, b, c = eq.groups()
    x, y, z = map(Fraction, (a, b, c))
    reverse = INVERSE[op]
    if (not y and op in "*/") or z != Fraction(gold):
        return None
    if OPS[op](x, y) != z or OPS[reverse](z, y) != x:
        return None
    clean = re.sub(r"<<[^<>]+>>", "", explanation).strip()
    return f"{clean}\nCheck: {c} {reverse} {b} = {a}.\n</think>\n\nFinal answer: {gold}"


def material(spec):
    previous = read(rationale_transfer.DATA)
    panel = rationale_transfer.material(spec, [r["question"] for r in previous])
    reserve = read("research/data/gsm-equations-01.json")
    used = {
        " ".join(r["question"].split())
        for r in previous
        + panel
        + reserve["train"]
        + reserve["dev"]
        + read("research/data/external-anchor.json")["gsm"]
    }
    hashed = set().union(
        *json.loads(
            gzip.decompress(
                Path("research/registry/historical-text-ids.json.gz").read_bytes()
            )
        ).values()
    )
    source = Path("data/research_external/gsm-train.jsonl")
    if digest(source) != reserve["source"]["sha256"]:
        raise ValueError("Publisher source changed")
    rows = [json.loads(line) for line in source.read_bytes().splitlines()]
    tok = load_tokenizer(local_snapshot())
    docs, selected = [], []
    for idx in sorted(range(len(rows)), key=lambda i: fingerprint([spec["seed"], i])):
        q = " ".join(rows[idx]["question"].split())
        if q in used or any(
            hashlib.sha256(("text:" + q + suffix).encode()).hexdigest() in hashed
            for suffix in (
                "",
                " Answer:",
                " Show your steps. End with Final answer: <number>. Solution:",
            )
        ):
            continue
        answer = checked_response(rows[idx]["answer"])
        if answer is None:
            continue
        prompt = tok.decode(prompt_ids(tok, {"question": q}, {"demos": 0}))
        doc = core.doc(tok, prompt, answer)
        if len(doc["ids"]) - 1 > spec["cap_positions"]:
            continue
        used.add(q)
        docs.append(doc)
        selected.append(
            {
                "source_row": idx,
                "question_id": fingerprint(q),
                "response_id": fingerprint(answer),
            }
        )
        if len(docs) == spec["updates"] * spec["logical_batch"]:
            break
    if len(docs) != spec["updates"] * spec["logical_batch"]:
        raise ValueError("Too few eligible independent training examples")
    return docs, selected, panel


def run(freeze=False):
    spec = read(REG)
    docs, selected, panel = material(spec)
    manifest = {
        "source": read("research/data/gsm-equations-01.json")["source"]["sha256"],
        "documents": fingerprint(docs),
        "selected": selected,
    }
    for path, obj in ((PANEL, panel), (TRAIN, manifest)):
        if freeze and not path.exists():
            save(path, obj)
        if read(path) != obj:
            raise ValueError(f"Frozen data differs: {path}")
    if freeze:
        return
    index = read("research/registry.json")
    for p in (PANEL, TRAIN):
        if digest(p) != index["datasets"][str(p)]["sha256"]:
            raise ValueError("Registered material differs")
    for path, key in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(spec[path]) / "adapters.safetensors") != spec[key]:
            raise ValueError("Paired parents differ")
    root = core.OUT / spec["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    core.assets()
    budget = core.Budget(spec["id"], spec["limits"])
    decision = {
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
        "passed": False,
    }
    cp = Path("checkpoints/research") / spec["id"]
    try:
        m, tok, parent = load_trained(spec["parent"])
        m.checkpointing = True
        state = core.train(
            m,
            docs,
            dict(spec, deployment=parent["spec"]["deployment"]),
            cp,
            budget,
            spec["parent_sha256"],
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        weight = digest(gen / "adapters.safetensors")
        decision.update(
            weights=weight,
            updates=state["step"],
            targets=sum(r["targets"] for r in state["history"]),
            update_seconds=sum(r["seconds"] for r in state["history"]),
        )
        del m, tok
        gc.collect()
        mx.clear_cache()
        rationale_transfer.evaluate_pair(
            root, root / "model", panel, spec, weight, budget, decision, retention=True
        )
        full, short, retention = (
            decision[k] for k in ("full", "prefix_256", "retention")
        )
        overall = full["all"]
        decision["passed"] = (
            overall["wins"] - overall["losses"] >= 2
            and overall["p"] < 0.05
            and overall["candidate"]["seconds"] <= overall["base"]["seconds"]
            and all(
                full[src]["wins"] >= full[src]["losses"]
                and short[src]["wins"] > short[src]["losses"]
                for src in spec["sources"]
            )
            and retention["candidate"]["nll"] <= retention["base"]["nll"] + 0.03
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        decision.update(status="interrupted", reason=str(exc))
        if (cp / "latest.json").exists():
            gen = checkpoint_location(cp)
            decision["saved_update"] = read(gen / "state.json")["step"]
            decision["optimizer"] = digest(gen / "optimizer.safetensors")
    finally:
        budget.close()
    save(root / "decision.json", decision)
    print(decision["status"], decision["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
