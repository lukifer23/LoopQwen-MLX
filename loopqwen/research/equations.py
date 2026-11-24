"""Full-vocabulary supervision and first-turn scoring of checked publisher calculations."""

import ast
import gc
import gzip
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from statistics import median

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_tokenizer

from . import core, bootstrap
from ..data.public_traces import value
from ..evaluation.answers import numeric
from ..evaluation.language7 import base, prompt_ids
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import read, save, digest, checkpoint_location
from ..model_loader import local_snapshot

NAME = "gsm-equations-01"
DATA = Path("research/data") / (NAME + ".json")


def equations(text):
    """Check arithmetic with a restricted exact rational interpreter."""
    checked = []
    for line in text.strip().splitlines():
        pieces = line.strip().split("=")
        if (
            len(pieces) != 2
            or len(line) > 160
            or not re.fullmatch(r"[0-9.,()+*/\s-]+", pieces[0])
        ):
            return []
        lhs, rhs = (s.strip().replace(",", "") for s in pieces)
        try:
            result = Fraction(rhs)
            if value(ast.parse(lhs, mode="eval").body) != result:
                return []
        except (
            ValueError,
            ZeroDivisionError,
            SyntaxError,
            OverflowError,
            RecursionError,
        ):
            return []
        checked.append(result)
    return checked


def ids(tok, item):
    return prompt_ids(
        tok,
        dict(
            question=item["question"] + "\nWrite each calculation as a short equation."
        ),
        dict(demos=1),
    )


def document(tok, item):
    prompt = tok.decode(ids(tok, item))
    response = (
        "\n".join(item["equations"]) + "\n</think>\n\nFinal answer: " + item["gold"]
    )
    return core.doc(tok, prompt, response)


def build():
    spec = read(Path("research/experiments") / (NAME + ".json"))
    src = Path("data/research_external/gsm-train.jsonl")
    if digest(src) != spec["source"]["sha256"]:
        raise ValueError("Publisher dataset differs")
    tok = load_tokenizer(local_snapshot())
    past = json.loads(
        gzip.decompress(
            Path("research/registry/historical-text-ids.json.gz").read_bytes()
        )
    )
    used = set().union(*(set(a) for a in past.values()))
    excluded = {r["id"] for r in read("research/data/external-anchor.json")["gsm"]}

    def seen(q):
        for form in (
            q,
            q + "\nAnswer:",
            q + "\nShow your steps. End with Final answer: <number>.\nSolution:",
        ):
            if (
                hashlib.sha256(("text:" + " ".join(form.split())).encode()).hexdigest()
                in used
            ):
                return True
        return False

    raw = [json.loads(l) for l in src.read_text().splitlines()]
    old_train = set()
    for f in Path("data").glob("*/train.jsonl"):
        for l in f.open():
            r = json.loads(l)
            old_train.add(
                r.get("question", r.get("prompt", "")).split("\nAnswer:")[0].strip()
            )
    unannot = [
        fingerprint(r["question"].strip())
        for r in raw
        if "<<" not in r["answer"] and r["question"].strip() not in old_train
    ]
    excluded.update(sorted(unannot, key=lambda i: fingerprint([71109, i]))[:12])
    data = {"train": [], "dev": []}
    for index, r in sorted(
        enumerate(raw),
        key=lambda ir: fingerprint([spec["seed"], ir[1]["question"].strip()]),
    ):
        q = r["question"].strip()
        key = fingerprint(q)
        if key in excluded or q in old_train or seen(q):
            continue
        gold = numeric(r["answer"].rsplit("####", 1)[-1])
        if gold is None or "/" in gold:
            continue
        item = dict(id=key, source_row=index, question=q, gold=gold)
        if "<<" not in r["answer"]:
            if len(data["dev"]) < spec["dev_maps"]:
                data["dev"].append(item)
                excluded.add(key)
        elif len(data["train"]) < spec["train_maps"]:
            steps = re.findall(r"<<([^<>]+)>>", r["answer"])
            valid = equations("\n".join(steps))
            if not 2 <= len(valid) <= 5 or valid[-1] != Fraction(gold):
                continue
            item["equations"] = steps
            doc = document(tok, item)
            if len(doc["ids"]) - 1 <= spec["cap_positions"]:
                data["train"].append(item)
                excluded.add(key)
        if (
            len(data["dev"]) == spec["dev_maps"]
            and len(data["train"]) == spec["train_maps"]
        ):
            break
    if (len(data["train"]), len(data["dev"])) != (spec["train_maps"], spec["dev_maps"]):
        raise ValueError("Insufficient verified disjoint examples")
    data.update(
        source=spec["source"], tokenizer=core.assets()["files"]["tokenizer.json"]
    )
    if DATA.exists() and read(DATA) != data:
        raise ValueError("Reserved examples changed")
    save(DATA, data)


def evaluate(m, t, weight, s, items, b, path):
    records = []
    for r in items:
        stream = ids(t, r)
        v = core.generate(m, t, stream, s["generation_cap"], b.check, thinking=True)
        answer = numeric(v["visible"]) if v["complete"] else None
        vals = (
            equations(t.decode(v["token_ids"][: v["close_at"]]))
            if v["close_at"] is not None
            else []
        )
        consistent = bool(vals and answer is not None and str(vals[-1]) == answer)
        records.append(
            dict(
                id=r["id"],
                answer=answer,
                gold=r["gold"],
                correct=answer == r["gold"],
                equations_consistent=consistent,
                **v,
            )
        )
    packed(path, dict(weights=weight, cohort=digest(DATA), records=records))
    return dict(
        n=len(records),
        valid=sum(r["answer"] is not None for r in records),
        correct=sum(r["correct"] for r in records),
        consistent_correct=sum(
            r["correct"] and r["equations_consistent"] for r in records
        ),
        tokens=sum(len(r["token_ids"]) for r in records),
        seconds=sum(r["seconds"] for r in records),
        median_ms=1000 * median(r["seconds"] for r in records),
    )


def run():
    s = read(Path("research/experiments") / (NAME + ".json"))
    out = core.OUT / NAME
    if core.archived_result(out, s):
        return
    if digest(DATA) != read("research/registry.json")["datasets"][str(DATA)]["sha256"]:
        raise ValueError("External split changed")
    data = read(DATA)
    b = core.Budget(NAME, s["limits"])
    try:
        par = Path(s["parent_path"])
        if digest(par / "adapters.safetensors") != s["parent_sha256"]:
            raise ValueError("Wrong trace parent")
        m, t, w = base()
        result = dict(base=evaluate(m, t, w, s, data["dev"], b, out / "base.json.gz"))
        del m, t
        gc.collect()
        mx.clear_cache()
        m, t = core.model(par, s["depth"])
        baseline = bootstrap.categorical(m, t, b)
        docs = [document(t, r) for r in data["train"]]
        cp = Path("checkpoints/research") / NAME
        st = core.train(m, docs, s, cp, b, s["parent_sha256"])
        gen = checkpoint_location(cp)
        w = digest(gen / "adapters.safetensors")
        if not (out / "model").exists():
            bootstrap.release(gen, out / "model")
        result["candidate"] = evaluate(
            m, t, w, s, data["dev"], b, out / "candidate.json.gz"
        )
        a, z = (
            unpack(out / (k + ".json.gz"))["records"] for k in ("base", "candidate")
        )
        if [r["prompt_ids"] for r in a] != [r["prompt_ids"] for r in z]:
            raise ValueError("Unpaired prompts")
        delta = np.array(
            [int(y["correct"]) - int(x["correct"]) for x, y in zip(a, z, strict=True)]
        )
        ix = np.random.default_rng(s["seed"]).integers(len(z), size=(5000, len(z)))
        after = bootstrap.categorical(m, t, b)
        result.update(
            paired=dict(
                wins=int((delta > 0).sum()),
                losses=int((delta < 0).sum()),
                ci=np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist(),
            ),
            anchor_before=baseline,
            anchor_after=after,
            nll={str(k): bootstrap.retention(m, k, b) for k in (1, 3)},
            weights=w,
            targets=sum(r["targets"] for r in st["history"]),
            update_seconds=sum(r["seconds"] for r in st["history"]),
            passed=bool(
                delta.sum() >= 2
                and result["candidate"]["valid"] >= 16
                and baseline[-1] - after[-1] <= 2
            ),
        )
        save(out / "decision.json", result)
    finally:
        b.close()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["data"]:
        build()
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit("Use data to freeze the cohort, or no argument to run")
