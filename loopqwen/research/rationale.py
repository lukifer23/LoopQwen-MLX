"""Evaluate the bounded Phase 4 language adaptation against untouched Base."""

import gc
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION, checkpoint
from ..evaluation.answers import numeric
from ..evaluation.language7 import base, prompt_ids
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import bootstrap, core

REG = Path("research/experiments/gsm-rationale-01.json")


def material(spec):
    t = load_tokenizer(local_snapshot())
    cohort = Path(spec["data"])
    expected = read("research/registry.json")["datasets"][str(cohort)]["sha256"]
    if digest(cohort) != expected:
        raise ValueError("Reserved GSM8K cohort changed")
    pool = read(cohort)
    source = Path("data/research_external/gsm-train.jsonl")
    if digest(source) != pool["source"]["sha256"]:
        raise ValueError("Publisher source changed")
    raw = [json.loads(line) for line in source.read_bytes().splitlines()]
    docs, selected = [], []
    for row in pool["train"]:
        publisher = raw[row["source_row"]]
        if publisher["question"].strip() != row["question"]:
            raise ValueError("Source question mismatch")
        rationale, gold = publisher["answer"].rsplit("####", 1)
        if numeric(gold) != row["gold"]:
            raise ValueError("Publisher target mismatch")
        explanation = re.sub(r"<<[^<>]+>>", "", rationale).strip()
        ids = prompt_ids(t, row, dict(demos=0))
        doc = core.doc(
            t, t.decode(ids), explanation + "\n</think>\n\nFinal answer: " + row["gold"]
        )
        if len(doc["ids"]) - 1 <= spec["cap_positions"]:
            docs.append(doc)
            selected.append(row["id"])
        if len(docs) == spec["updates"] * spec["logical_batch"]:
            break
    items = [dict(**r, prompt_ids=prompt_ids(t, r, dict(demos=0))) for r in pool["dev"]]
    if (
        len(set(selected + [r["id"] for r in items])) != len(docs) + len(items)
        or len(items) != 32
    ):
        raise ValueError("Cohort overlap or size mismatch")
    if len(docs) != spec["updates"] * spec["logical_batch"]:
        raise ValueError("Insufficient eligible training documents")
    manifest = dict(
        cohort=expected,
        train_ids=selected,
        train_tokens=fingerprint(docs),
        dev_ids=[r["id"] for r in items],
        dev_prompts=fingerprint([r["prompt_ids"] for r in items]),
    )
    return docs, items, manifest


def assess(m, t, weight, items, s, b, folder):
    rows = []
    for row in items:
        ids = row["prompt_ids"]
        if prompt_ids(t, row, dict(demos=0)) != ids:
            raise ValueError("Evaluation prompt IDs changed")
        stream = core.generate(m, t, ids, s["generation_cap"], b.check, thinking=True)
        answer = numeric(stream["visible"]) if stream["complete"] else None
        rows.append(
            dict(
                id=row["id"],
                gold=row["gold"],
                answer=answer,
                correct=answer == row["gold"],
                **stream,
            )
        )
        packed(folder, dict(weights=weight, records=rows))
    return dict(
        correct=sum(r["correct"] for r in rows),
        valid=sum(r["answer"] is not None for r in rows),
        tokens=sum(len(r["token_ids"]) for r in rows),
        seconds=sum(r["seconds"] for r in rows),
    )


def compare(model_path, weight, items, s, b, root, result):
    m, t, _ = checkpoint.load_trained(model_path)
    result["phase"] = phase = "candidate"
    m.eval()
    result[phase] = assess(m, t, weight, items, s, b, root / "candidate.json.gz")
    del m, t
    gc.collect()
    mx.clear_cache()
    result["phase"] = phase = "base"
    m, t, _ = base()
    result[phase] = assess(m, t, REVISION, items, s, b, root / "base.json.gz")
    data = [unpack(root / f"{a}.json.gz")["records"] for a in ("base", "candidate")]
    if [r["prompt_ids"] for r in data[0]] != [r["prompt_ids"] for r in data[1]]:
        raise ValueError("Unpaired development prompts")
    diff = [int(c["correct"]) - int(a["correct"]) for a, c in zip(*data, strict=True)]
    halves = [sum(diff[:16]), sum(diff[16:])]
    result.update(wins=diff.count(1), losses=diff.count(-1), half_deltas=halves)
    result["phase"] = "retention"
    result["base"]["retention"] = bootstrap.retention(m, 1, b)
    del m, t
    gc.collect()
    mx.clear_cache()
    m, _, _ = checkpoint.load_trained(model_path)
    result["candidate"]["retention"] = bootstrap.retention(m, s["depth"], b)
    c, a = result["candidate"], result["base"]
    result.update(
        status="completed",
        passed=bool(
            c["correct"] >= 8
            and min(halves) > 0
            and sum(diff) >= 4
            and c["seconds"] <= 1.25 * a["seconds"]
            and c["retention"]["nll"] <= a["retention"]["nll"] + 0.03
        ),
    )


def run(freeze=False):
    s = read(REG)
    docs, items, manifest = material(s)
    frozen = Path("research/data") / (s["id"] + ".json")
    if freeze:
        if frozen.exists() and read(frozen) != manifest:
            raise ValueError("Frozen cohort changed")
        save(frozen, manifest)
        return
    if manifest != read(frozen):
        raise ValueError("Material differs from registration")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    for path, key in [
        (s["parent"], "parent_sha256"),
        (s["dense_control"], "dense_sha256"),
    ]:
        if digest(Path(path) / "adapters.safetensors") != s[key]:
            raise ValueError("Phase 4 parent or dense control changed")
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    result = dict(registration=digest(REG), data=digest(frozen), passed=False)
    phase = "training"
    cp = Path("checkpoints/research") / s["id"]
    try:
        m, t, parent = checkpoint.load_trained(s["parent"])
        if m.span.passes != s["depth"]:
            raise ValueError("Registered depth differs")
        m.checkpointing = True
        deployment = dict(config=parent["config"], alpha_override=m.alpha_override)
        st = core.train(
            m, docs, dict(s, deployment=deployment), cp, b, s["parent_sha256"]
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        weight = digest(gen / "adapters.safetensors")
        result.update(
            weights=weight,
            updates=st["step"],
            targets=sum(r["targets"] for r in st["history"]),
            update_seconds=sum(r["seconds"] for r in st["history"]),
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        compare(root / "model", weight, items, s, b, root, result)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(
            status="interrupted", phase=result.get("phase", phase), reason=str(exc)
        )
        if (cp / "latest.json").exists():
            gen = checkpoint_location(cp)
            result.update(
                updates=read(gen / "state.json")["step"],
                generation=str(gen),
                weights=digest(gen / "adapters.safetensors"),
                optimizer=digest(gen / "optimizer.safetensors"),
            )
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result, flush=True)


def endpoint():
    reg = Path("research/experiments/gsm-rationale-endpoint.json")
    protocol = read(reg)
    spec = read(REG)
    _, items, manifest = material(spec)
    frozen = Path("research/data") / (spec["id"] + ".json")
    cp = checkpoint_location(Path("checkpoints/research") / spec["id"])
    weight = digest(cp / "adapters.safetensors")
    if (
        manifest != read(frozen)
        or weight != protocol["weights"]
        or read(cp / "state.json")["step"] != protocol["updates"]
    ):
        raise ValueError("Endpoint or evaluation cohort differs")
    root = core.OUT / protocol["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    core.assets()
    bootstrap.release(cp, root / "model")
    b = core.Budget(protocol["id"], protocol["limits"])
    result = dict(
        registration=digest(reg),
        data=digest(frozen),
        weights=weight,
        updates=protocol["updates"],
        passed=False,
    )
    try:
        compare(root / "model", weight, items, spec, b, root, result)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    result["descriptive_thresholds_met"] = result["passed"]
    result["passed"] = False
    save(root / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--endpoint", action="store_true")
    args = p.parse_args()
    if args.endpoint:
        endpoint()
    else:
        run(args.freeze)
