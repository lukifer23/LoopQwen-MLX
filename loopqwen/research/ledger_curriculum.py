"""Train a shared-block adapter and measure full-history lookup scaling."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base, interpret
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import bootstrap, core, rationale
from .ledger import ledger

REG = Path("research/experiments/ledger-curriculum-01.json")
DATA = Path("research/data/ledger-curriculum-01.json")


def material(s):
    t = load_tokenizer(local_snapshot())
    replay, arithmetic, parent_manifest = rationale.material(read(rationale.REG))
    if parent_manifest != read("research/data/gsm-rationale-01.json"):
        raise ValueError("Parent training or arithmetic evaluation changed")
    docs = []
    for j in range(s["updates"]):
        for k in range(3):
            i = 3 * j + k
            row = ledger(
                t, s["seed"] + i, (160, 288, 416)[k], (i * 37 % 97 + 1) / 99, i % 2 == 0
            )
            response = f"The entry for {row['key']} lists {row['gold']}.\n</think>\n\nFinal answer: {row['gold']}"
            doc = core.doc(t, t.decode(row["prompt_ids"]), response)
            if len(doc["ids"]) - 1 > s["cap_positions"]:
                raise ValueError("Ledger document exceeds cap")
            docs.append(doc)
        docs.append(replay[j])
    items = []
    for length in (2048, 16384):
        for frac in (0.10, 0.75):
            for announce in (False, True):
                items.append(
                    ledger(t, s["seed"] + 10000 + len(items), length, frac, announce)
                )
    manifest = {
        "training_ids": fingerprint(docs),
        "replay_ids": parent_manifest["train_ids"][: s["updates"]],
        "items": items,
    }
    return docs, arithmetic, manifest


def measure(m, t, row, cap, budget, *, offload_dir=None, reserve_full_kv=False):
    ids = row["prompt_ids"]
    end = core.atom(t, "<|im_end|>")
    text = t.decode(ids)
    if t.encode(text, add_special_tokens=False) != ids:
        raise ValueError("Evaluation token stream changed")
    v = generate(
        m,
        t,
        text,
        cap,
        progress=lambda _: budget.check(),
        stop_ids=[end],
        offload_dir=offload_dir,
        reserve_full_kv=reserve_full_kv,
    )
    del v["prompt"]
    stops = set(t.eos_token_ids) | {end}
    parsed = interpret(
        v["token_ids"], core.atom(t, "<think>"), core.atom(t, "</think>"), stops, t
    )
    answer = numeric(parsed["visible"]) if v["token_ids"][-1] in stops else None
    return dict(
        **v,
        ledger_seed=row["seed"],
        input_sha256=fingerprint(ids),
        gold=row["gold"],
        answer=answer,
        correct=answer == row["gold"],
    )


def scores(rows):
    return {
        "correct": sum(v["correct"] for v in rows),
        "seconds": sum(v["elapsed_s"] for v in rows),
    }


def run(freeze):
    s = read(REG)
    docs, arithmetic, manifest = material(s)
    if freeze and not DATA.exists():
        save(DATA, manifest)
    if read(DATA) != manifest:
        raise ValueError("Frozen training or ledger panel differs")
    if freeze:
        return
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    for path, key in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(s[path]) / "adapters.safetensors") != s[key]:
            raise ValueError("Registered parents differ")
    if digest(DATA) != read("research/registry.json")["datasets"][str(DATA)]["sha256"]:
        raise ValueError("Ledger registration changed")
    cp = Path("checkpoints/research") / s["id"]
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    result = {
        "passed": False,
        "qualified": [],
        "data": digest(DATA),
        "registration": digest(REG),
    }
    try:
        m, t, parent = load_trained(s["parent"])
        m.checkpointing = True
        state = core.train(
            m,
            docs,
            dict(s, deployment=parent["spec"]["deployment"]),
            cp,
            b,
            s["parent_sha256"],
        )
        gen = checkpoint_location(cp)
        published = root / "model"
        if published.exists():
            if any(
                digest(gen / name) != digest(published / name)
                for name in ("state.json", "adapters.safetensors")
            ) or read(published / "recovery.json")["optimizer_sha256"] != digest(
                gen / "optimizer.safetensors"
            ):
                raise ValueError(
                    "Existing ledger checkpoint differs from completed training"
                )
        else:
            bootstrap.release(gen, published)
        result.update(
            weights=digest(gen / "adapters.safetensors"),
            updates=state["step"],
            targets=sum(h["targets"] for h in state["history"]),
            update_seconds=sum(h["seconds"] for h in state["history"]),
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        reports = {}
        rows = manifest["items"]
        for arm in ("base", "candidate"):
            m, t, _ = base() if arm == "base" else load_trained(root / "model")
            reports[arm] = []
            for row in rows:
                reports[arm].append(measure(m, t, row, s["generation_cap"], b))
                packed(root / (arm + ".json.gz"), {"records": reports[arm]})
            del m, t
            gc.collect()
            mx.clear_cache()
        # Reuse exact Base generations and retention, digest-bound to the parent study.
        audit = {"data": digest(Path("research/data/gsm-rationale-01.json"))}
        rationale.compare(
            root / "model",
            result["weights"],
            arithmetic,
            read(rationale.REG),
            b,
            root / "arithmetic",
            audit,
            baseline_archive=Path(s["parent"]).parent,
        )
        result["arithmetic"] = audit
        result["scores"] = {}
        for announce, name in ((False, "query_last"), (True, "query_first")):
            group = {
                arm: scores(
                    [
                        rec
                        for rec, row in zip(report, rows, strict=True)
                        if row["announce"] == announce
                    ]
                )
                for arm, report in reports.items()
            }
            result["scores"][name] = group
            a, c = group["base"], group["candidate"]
            if (
                c["correct"] == 4
                and c["correct"] >= a["correct"]
                and c["seconds"] <= 1.75 * a["seconds"]
                and audit["candidate"]["retention"]["nll"]
                <= audit["base"]["retention"]["nll"] + 0.03
            ):
                result["qualified"].append(name)
        result.update(
            status="completed",
            passed=bool(result["qualified"]),
            arithmetic_retained=audit["candidate"]["correct"] >= 13,
        )
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
        if (cp / "latest.json").exists():
            gen = checkpoint_location(cp)
            result["saved_update"] = read(gen / "state.json")["step"]
            result["optimizer"] = digest(gen / "optimizer.safetensors")
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["qualified"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
