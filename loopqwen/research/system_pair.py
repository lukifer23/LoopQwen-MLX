"""Train an input-only depth selector, then test one deployable model across its workloads."""

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION
from ..data import retention3
from ..deployment import load_trained
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..evaluation.mixed import assess, summarize
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import (
    bootstrap,
    core,
    inference_data,
    memory_data,
    paired_solutions,
    route_training,
)


def material(s):
    t = load_tokenizer(local_snapshot())
    old = [r for path in s["excluded_panels"] for r in read(path)]
    known = {
        i for path in s["excluded_training"] for i in read(path)["ordered_questions"]
    }
    _, math = paired_solutions.material(
        dict(s, per_source=16, pool=0, rollout_cap=192), old, exclude_ids=known
    )
    panel = [
        {
            "id": r["id"],
            "prompt_ids": r["prompt_ids"],
            "family": "arithmetic" if i < 24 else "short_arithmetic",
            "source": r["source"],
            "cap": 1024 if i < 24 else 256,
            "gold": r["gold"],
        }
        for i, r in enumerate(math)
    ]
    past = [r for p in s["excluded_relations"] for r in read(p)]
    original = read("research/experiments/entailment-triplets-01.json")
    if "entailment" in s["experts"]:
        first = read("research/data/entailment-triplets-01.json")
        groups, _ = inference_data.training_groups(original, first)
        past += [r for source in groups.values() for group in source for r in group]
    nli = inference_data.material(
        dict(original, seed=s["seed"], per_source_groups=2, eval_template=3), past
    )
    panel.extend(
        {
            "id": r["id"],
            "family": "entailment",
            "source": r["source"],
            "cap": 1024,
            "prompt_ids": r["prompt_ids"],
            "gold": r["label"],
            "premise": r["premise"],
        }
        for r in nli
    )
    squad = read("data/research_external/squad-dev-v1.1.json")["data"]
    titles = set(s["excluded_titles"])
    documents = sorted(
        (a for a in squad if a["title"] not in titles),
        key=lambda a: fingerprint([s["seed"], a["title"]]),
    )
    for article in documents[:8]:
        ps = article["paragraphs"]
        p = min(ps, key=lambda p: fingerprint([s["seed"], p["context"]]))
        qa = min(p["qas"], key=lambda q: fingerprint([s["seed"], q["id"]]))
        passages = [p] + [other for other in ps if other["context"] != p["context"]][:4]
        passages.sort(key=lambda q: fingerprint([s["seed"], q["context"]]))
        text = (
            "\n\n".join(v["context"] for v in passages)
            + "\n\nQuestion: "
            + qa["question"]
            + "\nGive the shortest phrase from the article that answers the question."
        )
        panel.append(
            {
                "id": qa["id"],
                "family": "reading",
                "source": article["title"],
                "prompt_ids": route_training.chat(t, text),
                "cap": 256,
                "aliases": [a["text"] for a in qa["answers"]],
            }
        )
    for i in range(4):
        r = memory_data.make_item(
            t,
            s["seed"] + 11000 + i,
            (2048, 8192)[i // 2],
            (0.15, 0.75)[i % 2],
            ("filing", "statement")[i % 2],
        )
        panel.append(
            {
                "id": fingerprint(r["prompt_ids"]),
                "family": "register",
                "source": "register",
                "cap": 96,
                "prompt_ids": r["prompt_ids"],
                "gold": r["gold"],
                "gold_start": r["gold_start"],
                "gold_stop": r["gold_stop"],
            }
        )
    # Hash ordering is independent of gold, decoder selection and runtime.
    return sorted(panel, key=lambda r: fingerprint([s["seed"], r["id"]]))


def run(name, freeze):
    reg = Path("research/experiments") / (name + ".json")
    data = Path("research/data") / (name + ".json")
    train = Path("research/data") / (name + "-routing.json")
    s = read(reg)
    parent = Path(s["reader"]).parent
    if (
        digest(parent / "checksums.json") != s["reader_qualification"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Required register reader unavailable")
    if digest(Path(s["dense_control"]) / "adapters.safetensors") != s["dense_sha256"]:
        raise ValueError("Dense control changed")
    if not s["experts"]:
        raise ValueError(
            "A full-budget reasoning expert must qualify before this comparison"
        )
    for expert, info in s["experts"].items():
        route_training.qualified_expert(expert, info)
    t = load_tokenizer(local_snapshot())
    training = route_training.inputs(t, s)
    rows = material(s)
    manifest = {
        "documents": fingerprint(training),
        "ids": [fingerprint(x) for x in training],
        "labels": route_training.LABELS,
    }
    if set(manifest["ids"]) & {fingerprint(r["prompt_ids"]) for r in rows}:
        raise ValueError("Routing and deployment probes intersect")
    for p, obj in ((train, manifest), (data, rows)):
        if freeze and not p.exists():
            save(p, obj)
        if read(p) != obj:
            raise ValueError("Mixed comparison material changed")
    if freeze:
        return
    idx = read("research/registry.json")
    if idx["experiments"][name]["registration"] != digest(reg) or any(
        idx["datasets"][str(p)]["sha256"] != digest(p) for p in (train, data)
    ):
        raise ValueError("Unregistered conditional policy")
    root = core.OUT / name
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    b = core.Budget(name, s["limits"])
    result = {
        "passed": False,
        "registration": digest(reg),
        "data": digest(data),
        "training": digest(train),
    }
    try:
        m, t, _ = load_trained(s["reader"])
        mx.random.seed(s["seed"])
        w = route_training.package(s, m, t, training, root, b)
        result["weights"] = w
        del m, t
        gc.collect()
        mx.clear_cache()
        records = {"base": [], "candidate": []}
        ret = {}
        half = len(rows) // 2
        for offset, order in (
            (0, ("candidate", "base")),
            (half, ("base", "candidate")),
        ):
            for arm in order:
                m, t, _ = base() if arm == "base" else load_trained(root / "model")
                for r in rows[offset : offset + half]:
                    ids = r["prompt_ids"]
                    txt = t.decode(ids)
                    if t.encode(txt, add_special_tokens=False) != ids:
                        raise ValueError("Mixed prompt differs")
                    val = generate(
                        m,
                        t,
                        txt,
                        r["cap"],
                        progress=lambda _: b.check(),
                        stop_ids=(248046,),
                    )
                    del val["prompt"]
                    out = dict(
                        **val,
                        **assess(r, val["token_ids"], val["text"]),
                        id=r["id"],
                        input_sha256=fingerprint(ids),
                    )
                    records[arm].append(out)
                    packed(
                        root / (arm + ".json.gz"),
                        {
                            "weights": REVISION if arm == "base" else w,
                            "records": records[arm],
                        },
                    )
                if offset:
                    active = m if arm == "base" else m.context_policy.ordinary
                    if arm == "candidate":
                        retention3.build()
                        passages = [
                            json.loads(line)
                            for line in (retention3.ROOT / "validation.jsonl")
                            .read_text()
                            .splitlines()
                        ]
                        for raw in passages:
                            view, route = m.context_policy.choose(m, raw["ids"], 96)
                            if view is not active or route != "ordinary":
                                raise ValueError(
                                    "Raw retention request changed execution depth"
                                )
                        result["raw_route_ids"] = [r["id"] for r in passages]
                    ret[arm] = bootstrap.retention(active, 1, b)
                del m, t
                gc.collect()
                mx.clear_cache()
        scores = summarize(rows, records)
        a = scores["all"]
        preserved = all(
            c["token_ids"] == base_row["token_ids"]
            for c, base_row in zip(records["candidate"], records["base"], strict=True)
            if c["context_route"] == "ordinary"
        )
        fits = (
            a["wins"] - a["losses"] >= s["net_floor"]
            and all(v["wins"] >= v["losses"] for v in scores.values())
            and sum(v["wins"] > v["losses"] for k, v in scores.items() if k != "all")
            >= 2
            and sum(
                scores[k]["wins"] - scores[k]["losses"]
                for k in ("arithmetic", "entailment")
            )
            > 0
            and scores["register"]["candidate"]["correct"] == 4
            and a["candidate"]["seconds"] <= a["base"]["seconds"]
            and ret["base"] == ret["candidate"]
            and preserved
        )
        result.update(
            status="completed",
            scores=scores,
            retention=ret,
            ordinary_identity=preserved,
            passed=fits,
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    run(args.study, args.freeze)
