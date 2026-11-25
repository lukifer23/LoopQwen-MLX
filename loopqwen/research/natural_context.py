"""Train and evaluate a context-conditioned recurrent reader on fresh natural documents."""

import argparse
import gc
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION
from ..deployment import load_trained
from ..evaluation import extractive
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import core, natural_data, natural_fit, prose_reading

REG = Path("research/experiments/natural-context-01.json")
PANEL = Path("research/data/natural-context-01.json")
TRAIN = Path("research/data/natural-context-training.json")


def material(s):
    t = load_tokenizer(local_snapshot())
    train, docs = natural_data.make(t, s)
    panel = prose_reading.material(t, s)
    if {r["id"] for r in panel} & {r["id"] for r in train}:
        raise ValueError("Natural training and evaluation intersect")
    manifest = {
        "ids": [r["id"] for r in train],
        "passages": fingerprint(train),
        "documents": fingerprint(docs),
        "targets": sum(len(d["ids"]) - d["prefix"] for d in docs),
        "route_math": digest(Path(s["routing_math"])),
    }
    return train, docs, panel, manifest


def run(freeze):
    s = read(REG)
    train, docs, rows, manifest = material(s)
    for p, v in ((PANEL, rows), (TRAIN, manifest)):
        if freeze and not p.exists():
            save(p, v)
        if read(p) != v:
            raise ValueError("Frozen material differs")
    if freeze:
        return
    reg = read("research/registry.json")
    if reg["experiments"][s["id"]]["registration"] != digest(REG) or any(
        reg["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ):
        raise ValueError("Unregistered routed-reader trial")
    parent = Path(s["parent"])
    if (
        digest(parent.parent / "checksums.json") != s["qualification"]
        or not read(parent.parent / "decision.json")["passed"]
    ):
        raise ValueError("Original reader qualification failed")
    if digest(Path(s["dense_control"]) / "adapters.safetensors") != s["dense_sha256"]:
        raise ValueError("Dense control differs")
    root = core.OUT / s["id"]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    b = core.Budget(s["id"], s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
    }
    try:
        m, t, state = load_trained(parent)
        # Existing reader affects generation only. The prose adapter starts from the verified 32-update decoder.
        cp = Path("checkpoints/research") / s["id"]
        m.checkpointing = True
        st = core.train(
            m,
            docs,
            dict(s, deployment=state["spec"]["deployment"]),
            cp,
            b,
            digest(parent / "adapters.safetensors"),
        )
        modeldir = root / "model"
        shutil.copytree(parent, modeldir)
        gen = checkpoint_location(cp)
        shutil.copyfile(
            gen / "adapters.safetensors", modeldir / "prose-adapter.safetensors"
        )
        save(root / "prose-training.json", st)
        natural_fit.fit(s, m, t, train, docs, b, modeldir)
        files = {
            n: digest(modeldir / n)
            for n in (
                "attention-reader.json",
                "attention-reader.safetensors",
                "adapters.safetensors",
                "prose-adapter.safetensors",
                "prose-scorer.safetensors",
                "selector.safetensors",
            )
        }
        save(
            modeldir / "context-policy.json",
            {"files": files, "architecture": s["architecture"], "gain": s["gain"]},
        )
        result["weights"] = digest(modeldir / "context-policy.json")
        del m, t
        gc.collect()
        mx.clear_cache()
        reports = {"base": [], "candidate": []}
        half = len(rows) // 2
        for start, order in ((0, ("candidate", "base")), (half, ("base", "candidate"))):
            for arm in order:
                model, t, _ = base() if arm == "base" else load_trained(modeldir)
                for row in rows[start : start + half]:
                    ids = row["prompt_ids"]
                    txt = t.decode(ids)
                    if t.encode(txt, add_special_tokens=False) != ids:
                        raise ValueError("Input changed")
                    v = generate(
                        model,
                        t,
                        txt,
                        s["generation_cap"],
                        progress=lambda _: b.check(),
                        stop_ids=[core.atom(t, "<|im_end|>")],
                    )
                    del v["prompt"]
                    answer = extractive.answer(v["token_ids"], v["text"])
                    em, f1 = extractive.score(answer, row["aliases"])
                    section = v.get("attention_memory")
                    lo, hi = row["source_interval"]
                    reports[arm].append(
                        dict(
                            **v,
                            id=row["id"],
                            input_sha256=fingerprint(ids),
                            answer=answer,
                            correct=em,
                            f1=f1,
                            located=bool(
                                section
                                and lo < section["stop"]
                                and hi > section["start"]
                            ),
                        )
                    )
                    packed(
                        root / (arm + ".json.gz"),
                        {
                            "weights": REVISION if arm == "base" else result["weights"],
                            "records": reports[arm],
                        },
                    )
                del model, t
                gc.collect()
                mx.clear_cache()
        scores = {
            a: {
                "exact": sum(r["correct"] for r in rs),
                "f1": sum(r["f1"] for r in rs) / len(rs),
                "seconds": sum(r["elapsed_s"] for r in rs),
                "located": sum(r["located"] for r in rs),
            }
            for a, rs in reports.items()
        }
        aa, cc = scores["base"], scores["candidate"]
        result.update(
            status="completed",
            scores=scores,
            passed=cc["exact"] >= aa["exact"] + 3
            and cc["f1"] >= aa["f1"]
            and cc["seconds"] <= 1.25 * aa["seconds"],
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
