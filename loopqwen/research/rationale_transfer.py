"""Independent public-source confirmation with equal, longer thinking allowances."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION
from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import base, prompt_ids
from ..evaluation.mechanism import packed
from ..evaluation.statistics import exact_mcnemar
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import core

REG = Path("research/experiments/rationale-transfer-01.json")
DATA = Path("research/data/rationale-transfer-01.json")


def material(s, exclude=()):
    old = set().union(
        *json.loads(
            gzip.decompress(
                Path("research/registry/historical-text-ids.json.gz").read_bytes()
            )
        ).values()
    )
    prior = read("research/data/gsm-equations-01.json")
    used = {
        " ".join(r["question"].split())
        for r in prior["train"]
        + prior["dev"]
        + read("research/data/external-anchor.json")["gsm"]
    }
    used.update(" ".join(q.split()) for q in exclude)
    groups = {}
    for family, source in s["sources"].items():
        path = Path("data/research_external") / source["file"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(source["url"], timeout=45) as f:
                raw = f.read()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("Publisher source changed")
            path.write_bytes(raw)
        if digest(path) != source["sha256"]:
            raise ValueError("Publisher file digest mismatch")
        if family == "gsm1k_public":
            with path.open(newline="") as f:
                rows = [
                    (r["question"], r["answer"], family, i)
                    for i, r in enumerate(csv.DictReader(f))
                ]
        else:
            rows = [
                (
                    r["Body"].strip() + " " + r["Question"].strip(),
                    str(r["Answer"]),
                    r["Type"].replace("Common-Divison", "Common-Division"),
                    r["ID"],
                )
                for r in read(path)
            ]
        strata = {}
        for question, raw_gold, kind, source_id in rows:
            question = " ".join(question.split())
            gold = numeric(raw_gold)
            if question in used or any(
                hashlib.sha256(("text:" + question + suffix).encode()).hexdigest()
                in old
                for suffix in (
                    "",
                    " Answer:",
                    " Show your steps. End with Final answer: <number>. Solution:",
                )
            ):
                continue
            if gold is None:
                raise ValueError("Non-numeric publisher target")
            used.add(question)
            strata.setdefault(kind, []).append(
                {
                    "id": fingerprint(question),
                    "question": question,
                    "gold": gold,
                    "source": family,
                    "source_id": source_id,
                    "stratum": kind,
                }
            )
        if family == "svamp" and len(strata) != 4:
            raise ValueError("SVAMP operation strata changed")
        order = lambda r: fingerprint([s["seed"], r["id"]])
        per = s["per_source"] // len(strata)
        groups[family] = sorted(
            [r for rows in strata.values() for r in sorted(rows, key=order)[:per]],
            key=order,
        )
        if len(groups[family]) != s["per_source"]:
            raise ValueError("Insufficient evaluation rows")
    t = load_tokenizer(local_snapshot())
    return [
        dict(**r, prompt_ids=prompt_ids(t, r, {"demos": 0}))
        for pair in zip(*groups.values(), strict=True)
        for r in pair
    ]


def summarize(data, base_rows, candidate_rows, *, cap):
    def stats(rows):
        return {
            "correct": sum(r["correct"] and len(r["token_ids"]) <= cap for r in rows),
            "valid": sum(
                r["answer"] is not None and len(r["token_ids"]) <= cap for r in rows
            ),
            "tokens": sum(min(len(r["token_ids"]), cap) for r in rows),
            "seconds": sum(
                r["seconds_at_256"] if cap == 256 else r["seconds"] for r in rows
            ),
        }

    if [r["prompt_ids"] for r in base_rows] != [r["prompt_ids"] for r in data] or [
        r["prompt_ids"] for r in candidate_rows
    ] != [r["prompt_ids"] for r in data]:
        raise ValueError("Unpaired transfer records")
    out = {}
    for name in ("all", *sorted({r["source"] for r in data})):
        aa, cc = (
            [
                r
                for r, item in zip(rows, data, strict=True)
                if name == "all" or item["source"] == name
            ]
            for rows in (base_rows, candidate_rows)
        )
        diff = [
            int(c["correct"] and len(c["token_ids"]) <= cap)
            - int(a["correct"] and len(a["token_ids"]) <= cap)
            for a, c in zip(aa, cc, strict=True)
        ]
        wins, losses = diff.count(1), diff.count(-1)
        out[name] = {
            "base": stats(aa),
            "candidate": stats(cc),
            "wins": wins,
            "losses": losses,
            "n": len(diff),
            "p": exact_mcnemar(wins, losses),
        }
    return out


def evaluate_pair(root, model, rows, s, weight, b, result, *, retention=False):
    records = {"base": [], "candidate": []}
    for start, order in [
        (0, ("candidate", "base")),
        (len(rows) // 2, ("base", "candidate")),
    ]:
        for arm in order:
            m, t, _ = base() if arm == "base" else load_trained(model)
            for item in rows[start : start + len(rows) // 2]:
                if item["prompt_ids"] != prompt_ids(t, item, {"demos": 0}):
                    raise ValueError("Tokenizer mismatch")
                # The guard runs before each token, after the previous argmax synchronized.
                calls = 0
                at_256 = None
                first = None

                def guard():
                    nonlocal calls, at_256, first
                    if calls == 0:
                        first = perf_counter()
                    if calls == 256:
                        at_256 = perf_counter() - first
                    calls += 1
                    b.check()

                v = core.generate(
                    m,
                    t,
                    item["prompt_ids"],
                    s["generation_cap"],
                    guard,
                    thinking=True,
                )
                answer = numeric(v["visible"]) if v["complete"] else None
                records[arm].append(
                    dict(
                        **v,
                        id=item["id"],
                        gold=item["gold"],
                        answer=answer,
                        correct=answer == item["gold"],
                        seconds_at_256=at_256 if at_256 is not None else v["seconds"],
                    )
                )
                packed(
                    root / (arm + ".json.gz"),
                    {
                        "weights": REVISION if arm == "base" else weight,
                        "records": records[arm],
                    },
                )
            if retention and start:
                from .bootstrap import retention as measure_retention

                result.setdefault("retention", {})[arm] = measure_retention(
                    m, 1 if arm == "base" else 2, b
                )
            del m, t
            gc.collect()
            mx.clear_cache()
    full = summarize(
        rows, records["base"], records["candidate"], cap=s["generation_cap"]
    )
    result.update(
        status="completed",
        full=full,
        prefix_256=summarize(rows, records["base"], records["candidate"], cap=256),
    )


def run(freeze):
    s = read(REG)
    parent = Path(s["prerequisite"])
    if (
        not read(parent / "decision.json")["passed"]
        or digest(parent / "model/adapters.safetensors") != s["weights"]
    ):
        raise ValueError("Development prerequisite failed")
    rows = material(s)
    if freeze:
        if DATA.exists() and read(DATA) != rows:
            raise ValueError("Frozen panel changed")
        save(DATA, rows)
        return
    if (
        read(DATA) != rows
        or digest(DATA)
        != read("research/registry.json")["datasets"][str(DATA)]["sha256"]
    ):
        raise ValueError("Frozen transfer cohort changed")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "data": digest(DATA),
        "weights": s["weights"],
    }
    try:
        evaluate_pair(root, parent / "model", rows, s, s["weights"], b, result)
        full = result["full"]
        result["passed"] = (
            all(full[name]["wins"] - full[name]["losses"] >= 2 for name in s["sources"])
            and full["all"]["p"] < 0.05
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
        )
    except (InterruptedError, MemoryError, TimeoutError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
