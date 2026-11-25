"""Expand a recurrent arithmetic adapter across the decoder and test longer-budget reasoning."""

import argparse
import gc
import json
import re
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from .. import checkpoint
from ..evaluation.answers import numeric
from ..evaluation.language7 import prompt_ids
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import bootstrap, core, paired_solutions, rationale_transfer

REG = Path("research/experiments/all-layer-reasoning-01.json")
PANEL = Path("research/data/all-layer-reasoning-01.json")
TRAIN = Path("research/data/all-layer-reasoning-training.json")


def material(s):
    prior_paths = (
        paired_solutions.PANEL,
        paired_solutions.POOL,
        Path("research/data/shared-budget-vote-01.json"),
        *map(Path, s.get("exclude_panels", [])),
    )
    prior = [item for path in prior_paths for item in read(path)]
    prior_ids = {
        key
        for path in s.get("exclude_training", [])
        for key in read(path)["ordered_questions"]
    }
    _, panel = paired_solutions.material(s, prior, exclude_ids=prior_ids)
    excluded = {fingerprint(" ".join(r["question"].split())) for r in prior + panel}
    excluded.update(prior_ids)
    for name in ("rationale-transfer-01", "checked-rationale-01"):
        excluded.update(
            fingerprint(" ".join(r["question"].split()))
            for r in read(f"research/data/{name}.json")
        )
    old = read("research/data/gsm-equations-01.json")
    for r in (
        old["train"] + old["dev"] + read("research/data/external-anchor.json")["gsm"]
    ):
        excluded.add(fingerprint(" ".join(r["question"].split())))
    excluded.update(
        r["question_id"]
        for r in read("research/data/checked-rationale-training.json")["selected"]
    )
    t = load_tokenizer(local_snapshot())
    math_docs = []
    source = Path("data/research_external/gsm-train.jsonl")
    if digest(source) != s["sources"]["gsm8k_reserved"]["sha256"]:
        raise ValueError("Arithmetic source changed")
    entries = [json.loads(x) for x in source.read_bytes().splitlines()]
    for i in sorted(
        range(len(entries)), key=lambda i: fingerprint([s["seed"], "train", i])
    ):
        q = entries[i]
        key = fingerprint(" ".join(q["question"].split()))
        if key in excluded:
            continue
        explanation, raw = q["answer"].rsplit("####", 1)
        gold = numeric(raw)
        if gold is None:
            continue
        ids = prompt_ids(t, q, {"demos": 0})
        doc = core.doc(
            t,
            t.decode(ids),
            re.sub(r"<<[^<>]+>>", "", explanation).strip()
            + "\n</think>\n\nFinal answer: "
            + gold,
        )
        if len(doc["ids"]) - 1 <= s["cap_positions"]:
            math_docs.append((key, doc))
            excluded.add(key)
        if len(math_docs) == 3 * s["updates"]:
            break
    source = Path("data/research_external/squad-train-v1.1.json")
    if not source.exists():
        with urlopen(s["reading_url"], timeout=45) as f:
            raw = f.read()
        import hashlib

        if hashlib.sha256(raw).hexdigest() != s["reading_sha256"]:
            raise ValueError("Public reading download changed")
        source.write_bytes(raw)
    if digest(source) != s["reading_sha256"]:
        raise ValueError("Reading source changed")
    candidates = [
        (q, p["context"])
        for a in read(source)["data"]
        if a["title"] in s["reading_topics"]
        for p in a["paragraphs"]
        for q in p["qas"]
    ]
    reading_docs = []
    for q, passage in sorted(
        candidates, key=lambda v: fingerprint([s["seed"], v[0]["id"]])
    ):
        if q["id"] in prior_ids:
            continue
        ann = q["answers"][0]
        start = ann["answer_start"]
        answer = ann["text"]
        if passage[start : start + len(answer)] != answer:
            continue
        prefix = passage[:start]
        suffix = passage[start + len(answer) :]
        sentence = (
            re.split(r"(?<=[.!?])\s+", prefix)[-1]
            + answer
            + re.split(r"(?<=[.!?])\s+", suffix)[0]
        )
        user = (
            passage
            + "\n\nQuestion: "
            + q["question"]
            + "\nReason briefly. Give just the shortest phrase from the article that answers the question."
        )
        prompt = t.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        d = core.doc(t, prompt, sentence.strip() + "\n</think>\n\n" + answer)
        if len(d["ids"]) - 1 <= s["cap_positions"]:
            reading_docs.append((q["id"], d))
        if len(reading_docs) == s["updates"]:
            break
    if (len(math_docs), len(reading_docs)) != (3 * s["updates"], s["updates"]):
        raise ValueError("Incomplete training cohort")
    schedule = []
    for i, r in enumerate(reading_docs):
        schedule.extend([*math_docs[3 * i : 3 * i + 3], r])
    docs = [d for _, d in schedule]
    return (
        docs,
        panel,
        {
            "ordered_questions": [key for key, _ in schedule],
            "documents": fingerprint(docs),
            "response_tokens": sum(len(d["ids"]) - d["prefix"] for d in docs),
        },
    )


def run(freeze):
    s = read(REG)
    docs, rows, manifest = material(s)
    for p, v in ((TRAIN, manifest), (PANEL, rows)):
        if freeze and not p.exists():
            save(p, v)
        if read(p) != v:
            raise ValueError("Frozen material differs")
    if freeze:
        return
    for p, key in (
        (s["parent"], "parent_sha256"),
        (s["dense_control"], "dense_sha256"),
    ):
        if digest(Path(p) / "adapters.safetensors") != s[key]:
            raise ValueError("Parent or matched control differs")
    registry = read("research/registry.json")
    if any(
        registry["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ) or registry["experiments"][s["id"]]["registration"] != digest(REG):
        raise ValueError("Unregistered study")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    b = core.Budget(s["id"], s["limits"])
    result = {
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
        "passed": False,
    }
    cp = Path("checkpoints/research") / s["id"]
    try:
        parent = Path(s["parent"])
        dep = read(parent / "state.json")["spec"]["deployment"]
        dep = dict(
            dep,
            config=dict(
                dep["config"], target="all", seed=s["seed"], gradient_checkpointing=True
            ),
        )
        m, t, _ = checkpoint.setup(dep["config"])
        m.alpha_override = dep["alpha_override"]
        prior = mx.load(str(parent / "adapters.safetensors"))
        initialized = dict(tree_flatten(m.trainable_parameters()))
        if not prior.keys() < initialized.keys() or any(
            prior[k].shape != initialized[k].shape for k in prior
        ):
            raise ValueError("Expanded adapter cannot reproduce the parent")
        if any(
            mx.max(mx.abs(v)).item() != 0
            for k, v in initialized.items()
            if k not in prior and k.endswith("lora_b")
        ):
            raise ValueError("New projection must start at zero")
        m.load_weights(list(prior.items()), strict=False)
        del prior, initialized
        st = core.train(m, docs, dict(s, deployment=dep), cp, b, s["parent_sha256"])
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        result.update(
            weights=digest(gen / "adapters.safetensors"),
            updates=st["step"],
            targets=sum(v["targets"] for v in st["history"]),
            update_seconds=sum(v["seconds"] for v in st["history"]),
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        rationale_transfer.evaluate_pair(
            root, root / "model", rows, s, result["weights"], b, result, retention=True
        )
        full, short = result["full"], result["prefix_256"]
        result["passed"] = (
            full["all"]["wins"] - full["all"]["losses"] >= 4
            and all(
                full[k]["wins"] > full[k]["losses"]
                and short[k]["wins"] >= short[k]["losses"]
                for k in s["sources"]
            )
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
            and result["retention"]["candidate"]["nll"]
            <= result["retention"]["base"]["nll"] + 0.03
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    run(parser.parse_args().freeze)
