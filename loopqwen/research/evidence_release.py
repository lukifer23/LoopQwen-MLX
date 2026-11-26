"""Qualify the shipping evidence-loop checkpoint through the public native decoder."""

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from ..context_policy import features
from ..data import retention3
from ..deployment import load_trained
from ..evaluation.generation_eval import generate
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from . import bootstrap, core, ledger_curriculum, route_training

NAME = "evidence-language-release-01"


def package(s, t, model, root, b):
    parent = Path(s["combined"]) / "model"
    folder = root / "model"
    shutil.copytree(parent, folder)
    extra = folder / "evidence"
    extra.mkdir()
    for file in ("reader.json", "reader.safetensors"):
        shutil.copyfile(Path(s["reader"]) / file, extra / file)
    rows = read(s["material"])["train"]
    positives = [r["prompt_ids"] for r in rows]
    negatives = route_training.inputs(
        t, read(Path(s["combined"]) / "registration.json")
    )
    examples = positives + negatives
    x = []
    for ids in examples:
        b.check()
        x.append(features(model.text.model.embed_tokens, ids))
        mx.eval(x[-1])
    x = mx.stack(x)
    y = mx.array([1] * len(positives) + [0] * len(negatives))
    mx.random.seed(s["seed"])
    gate = nn.Linear(x.shape[1], 2)
    opt = optim.Adam(0.003)
    vg = nn.value_and_grad(
        gate, lambda net: nn.losses.cross_entropy(net(x), y, reduction="mean")
    )
    for _ in range(160):
        b.check()
        v, g = vg(gate)
        opt.update(gate, g)
        mx.eval(v, gate.parameters(), opt.state)
    gate.save_weights(str(extra / "scope.safetensors"))
    save(
        extra / "scope.json",
        {
            "depth": 2,
            "confidence": 0.99,
            "train_inputs": fingerprint(examples),
            "positive": len(positives),
            "negative": len(negatives),
            "example": t.decode(positives[0]),
        },
    )
    meta = read(folder / "context-policy.json")
    meta["schema"] = 3
    meta["files"].update(
        {str(f.relative_to(folder)): digest(f) for f in extra.iterdir()}
    )
    save(folder / "context-policy.json", meta)
    return folder


def run():
    reg = Path("research/experiments") / (NAME + ".json")
    s = read(reg)
    idx = read("research/registry.json")
    if idx["experiments"][NAME]["registration"] != digest(reg):
        raise ValueError("Unregistered release")
    for name, h in s["prerequisites"].items():
        arc = Path("research/results") / name
        if (
            digest(arc / "checksums.json") != h
            or not read(arc / "decision.json")["passed"]
        ):
            raise ValueError("Failed or changed prerequisite: " + name)
    b = core.Budget(NAME, s["limits"])
    root = core.OUT / NAME
    root.mkdir(exist_ok=False, parents=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    res = {"registration": digest(reg), "passed": False}
    try:
        parent, t, _ = load_trained(Path(s["combined"]) / "model")
        folder = package(s, t, parent, root, b)
        del parent
        gc.collect()
        mx.clear_cache()
        model, t, _ = load_trained(folder)
        res["policy"] = digest(folder / "context-policy.json")
        suites = [
            (
                "deduction",
                read(s["material"])["replication"],
                Path(s["longbase"]) / "recurrent.json.gz",
            ),
            (
                "mixed",
                read("research/data/conditional-language-02.json"),
                Path(s["combined"]) / "candidate.json.gz",
            ),
        ]
        for kind, rows, ref in suites:
            baseline = unpack(ref)["records"]
            path = root / (kind + ".json.gz")
            samples = unpack(path)["records"] if path.exists() else []
            for i, (row, expected) in enumerate(zip(rows, baseline, strict=True)):
                if i < len(samples):
                    continue
                b.check()
                text = t.decode(row["prompt_ids"])
                if t.encode(text, add_special_tokens=False) != row["prompt_ids"]:
                    raise ValueError("Prompt round-trip")
                v = generate(
                    model,
                    t,
                    text,
                    row.get("cap", 768),
                    stop_ids=(248046,),
                    progress=lambda _: b.check(),
                )
                route = (
                    "deduction" if kind == "deduction" else expected["context_route"]
                )
                samples.append(
                    {
                        "id": row["id"],
                        "route": v["context_route"],
                        "token_ids": v["token_ids"],
                        "elapsed_s": v["elapsed_s"],
                        "input_sha256": fingerprint(row["prompt_ids"]),
                    }
                )
                packed(path, {"records": samples})
                if (
                    v["context_route"] != route
                    or v["token_ids"] != expected["token_ids"]
                ):
                    raise ValueError(f"{kind} replay differs at {row['id']}")
            res[kind] = {
                "identical": len(samples),
                "seconds": sum(r["elapsed_s"] for r in samples),
            }
        retention3.build()
        rows = [
            json.loads(line)
            for line in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
        ]
        for r in rows:
            active, route = model.context_policy.choose(model, r["ids"], 96)
            if route != "ordinary" or active is not model.context_policy.ordinary:
                raise ValueError("Raw language route changed")
        ret = bootstrap.retention(model.context_policy.ordinary, 1, b)
        expected = read(Path(s["combined"]) / "decision.json")["retention"]["base"]
        if ret != expected:
            raise ValueError("Prose retention changed")
        res.update(retention=ret, raw_requests=len(rows))
        ref = Path(s["context"])
        docs = read("research/data/conditional-language-128k.json")["panel"]
        previous = unpack(ref / "candidate.json.gz")["records"]
        path = root / "context.json.gz"
        samples = unpack(path)["records"] if path.exists() else []
        for i, (row, v) in enumerate(zip(docs, previous, strict=True)):
            if i < len(samples):
                continue
            item = ledger_curriculum.measure(model, t, row, 96, b)
            samples.append(item)
            packed(path, {"records": samples})
            if (
                not item["correct"]
                or item["context_route"] != v["context_route"]
                or item["token_ids"] != v["token_ids"]
                or item["prompt_tokens"] < 131072
                or not (
                    item["attention_memory"]["start"]
                    <= row["gold_start"]
                    < row["gold_stop"]
                    <= item["attention_memory"]["stop"]
                )
            ):
                raise ValueError("Full-history replay differs")
        res["context"] = ledger_curriculum.scores(samples)
        res.update(
            status="completed",
            passed=res["context"]["seconds"]
            <= 1.75 * read(ref / "decision.json")["scores"]["base"]["seconds"],
        )
    except (MemoryError, TimeoutError, InterruptedError, ValueError) as exc:
        res.update(
            status="interrupted" if not isinstance(exc, ValueError) else "failed",
            reason=str(exc),
        )
    finally:
        b.close()
    save(root / "decision.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    run()
