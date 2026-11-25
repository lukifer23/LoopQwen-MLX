"""Train a Base-derived recurrent student against sparse teacher distributions and clean prose anchors."""

import argparse
import gc
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load as load_model
from mlx_lm.utils import load_tokenizer

from ..evaluation.answers import numeric
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..looped_model import LoopedModel
from ..model_loader import local_snapshot
from ..recurrence import Span
from . import all_layer, bootstrap, core, distill_training, rationale_transfer

REG = Path("research/experiments/projected-distillation-01.json")
PANEL = Path("research/data/projected-distillation-01.json")
TRAIN = Path("research/data/projected-distillation-training.json")


def anchors(t, s):
    groups = []
    for name, meta in s["anchor_files"].items():
        p = Path("data/research_external/distillation-prose") / name
        if not p.exists():
            p.parent.mkdir(exist_ok=True, parents=True)
            with urlopen(meta["url"], timeout=45) as f:
                raw = f.read()
            import hashlib

            if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError("Anchor download differs")
            p.write_bytes(raw)
        if digest(p) != meta["sha256"]:
            raise ValueError("Anchor source differs")
        ids = t.encode(p.read_text(), add_special_tokens=False)
        groups.append([ids[i : i + 128] for i in range(0, len(ids) - 127, 128)])
    novel = Path("data/v2/language_train.jsonl")
    if digest(novel) != s["novel_sha256"]:
        raise ValueError("Literary source differs")
    import json

    groups.append(
        [
            json.loads(l)["variants"]["language"]["ids"]
            for l in novel.read_bytes().splitlines()
        ]
    )
    for j, group in enumerate(groups):
        group.sort(key=lambda ids: fingerprint([s["seed"], j, ids]))
    options = [
        (j, i, ids)
        for i in range(max(map(len, groups)))
        for j, g in enumerate(groups)
        for ids in g[i : i + 1]
    ]
    if len(options) < s["updates"]:
        raise ValueError("Insufficient distinct prose passages")
    return [
        {"raw": True, "ids": ids, "prefix": 1} for _, _, ids in options[: s["updates"]]
    ]


def material(s):
    supervised, panel, parent_manifest = all_layer.material(s)
    plain = anchors(load_tokenizer(local_snapshot()), s)
    docs = []
    for i, anchor in enumerate(plain):
        docs.extend(dict(d, raw=False) for d in supervised[4 * i : 4 * i + 4])
        docs.append(anchor)
    tokenizer = load_tokenizer(local_snapshot())
    exposure = []
    for j, d in enumerate(docs):
        n = d["prefix"]
        label = None
        if j % s["group"] < 3:
            label = numeric(tokenizer.decode(d["ids"][n:-1]).split("</think>")[-1])
        exposure.append(
            {
                "prefix": n,
                "length": len(d["ids"]),
                "prompt": fingerprint(d["ids"][:n]),
                "raw": d["raw"],
                "gold": label,
            }
        )
    return (
        docs,
        panel,
        dict(parent_manifest, anchors=fingerprint(plain), exposure=exposure),
    )


def tutor(s):
    ref = s["teacher"]
    folder_path = Path(
        snapshot_download(ref["repo"], revision=ref["revision"], local_files_only=True)
    )
    if digest(folder_path / ref["shard"]) != ref["sha256"]:
        raise ValueError("Teacher weights differ")
    import json

    student_vocab = json.loads((local_snapshot() / "tokenizer.json").read_bytes())[
        "model"
    ]["vocab"]
    active_vocab = json.loads((folder_path / "tokenizer.json").read_bytes())["model"][
        "vocab"
    ]
    if (
        student_vocab != active_vocab
        or read(folder_path / "config.json")["text_config"]
        != read(local_snapshot() / "config.json")["text_config"]
    ):
        raise ValueError("Teacher lexical IDs or model configuration differs")
    outer, t = load_model(str(folder_path))
    model = LoopedModel(outer.language_model, Span(passes=1))
    del outer
    if any(
        t.encode(x, add_special_tokens=False) != [n]
        for x, n in (("<think>", 248068), ("</think>", 248069), ("<|im_end|>", 248046))
    ):
        raise ValueError("Teacher control-token identities differ")
    model.eval()
    return model, t


def verified_traces(s, docs, b, root):
    """Sample full reasoning with the teacher on training items only; preserve every attempt."""
    path = root / "training-draws.json.gz"
    draws = unpack(path)["records"] if path.exists() else []
    eligible = [i for i in range(len(docs)) if i % s["group"] < 3]
    eligible.sort(key=lambda i: fingerprint([s["seed"], "teacher", i]))
    model, t = tutor(s)
    selected = []
    for index in eligible[: s["max_draws"]]:
        doc = docs[index]
        gold = numeric(t.decode(doc["ids"][doc["prefix"] : -1]).split("</think>")[-1])
        if gold is None:
            raise ValueError("Publisher target missing")
        j = len(selected)
        if j < len(draws):
            r = draws[j]
        else:
            r = core.generate(
                model,
                t,
                doc["ids"][: doc["prefix"]],
                s["draw_cap"],
                b.check,
                thinking=True,
                seed=s["seed"] + j,
            )
            ans = numeric(r["visible"]) if r["complete"] else None
            r.update(index=index, gold=gold, answer=ans, correct=ans == gold)
            draws.append(r)
            packed(path, {"records": draws})
        if (
            r["index"] != index
            or r["prompt_ids"] != doc["ids"][: doc["prefix"]]
            or r["gold"] != gold
        ):
            raise ValueError("Teacher trajectory identity differs")
        selected.append(r)
        if sum(v["correct"] for v in selected) >= s["wanted"]:
            break
    good = [r for r in selected if r["correct"]]
    del model, t
    gc.collect()
    mx.clear_cache()
    if len(good) < s["min_good"]:
        return None, len(good)
    # Reserve distinct replacement slots, avoiding every drawn question's original slot.
    positions = [i for i in eligible if i not in {r["index"] for r in selected}]
    replacement = []
    for r in good:
        trajectory = {
            "ids": r["prompt_ids"] + r["token_ids"],
            "prefix": len(r["prompt_ids"]),
            "raw": False,
        }
        docs[r["index"]] = trajectory
        for rep in range(s["copies"] - 1):
            i = positions.pop()
            docs[i] = trajectory
            replacement.append([i, r["index"]])
    save(
        root / "draw-selection.json",
        {
            "indices": [r["index"] for r in good],
            "additional_slots": replacement,
            "documents": fingerprint(docs),
        },
    )
    return docs, len(good)


def collect(s, docs, b, folder):
    path = folder / "teachers.safetensors"
    if path.exists():
        if read(folder / "teachers.json")["sha256"] != digest(path):
            raise ValueError("Stored teacher targets differ")
        return path
    from time import perf_counter

    begin = perf_counter()
    arrays = {}
    for raw in (False, True):
        if raw:
            model, t, _ = base()
        else:
            model, t = tutor(s)
        model.eval()
        for j, d in enumerate(docs):
            if d["raw"] != raw:
                continue
            b.check()
            ids, p = distill_training.distribution(
                model, d, raw, s["positions"], s["top_k"], s["temperature"]
            )
            arrays[f"{j}.ids"] = mx.contiguous(ids.astype(mx.int32))
            mx.eval(arrays[f"{j}.ids"])
            arrays[f"{j}.mass"] = p
        del model, t
        gc.collect()
        mx.clear_cache()
    folder.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name("teachers-pending.safetensors")
    mx.save_safetensors(str(tmp), arrays)
    tmp.replace(path)
    save(
        folder / "teachers.json",
        {
            "sha256": digest(path),
            "seconds": perf_counter() - begin,
            "rows": len(docs),
            "positions": sum(
                v.shape[0] for k, v in arrays.items() if k.endswith(".ids")
            ),
        },
    )
    return path


def run(freeze):
    s = read(REG)
    docs, panel, manifest = material(s)
    for p, v in ((PANEL, panel), (TRAIN, manifest)):
        if freeze and not p.exists():
            save(p, v)
        if read(p) != v:
            raise ValueError("Frozen distillation inputs differ")
    if freeze:
        return
    registry = read("research/registry.json")
    if registry["experiments"][s["id"]]["registration"] != digest(REG) or any(
        registry["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ):
        raise ValueError("Unregistered inputs")
    for path, k in (
        (s["parent"], "parent_sha256"),
        (s["dense_control"], "dense_sha256"),
    ):
        if digest(Path(path) / "adapters.safetensors") != s[k]:
            raise ValueError("Wrong controls")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    core.assets()
    root.mkdir(exist_ok=True, parents=True)
    b = core.Budget(s["id"], s["limits"])
    cp = Path("checkpoints/research") / s["id"]
    result = {
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
        "passed": False,
    }
    try:
        docs, count = verified_traces(s, docs, b, root)
        result["correct_training_draws"] = count
        if docs is None:
            result.update(status="insufficient_correct_training_traces")
            save(root / "decision.json", result)
            return
        target = collect(s, docs, b, root)
        result["teacher_targets"] = digest(target)
        from .task_vectors import initialize

        model, t, dep = initialize(s, root, b, tutor)
        st = distill_training.train(
            model, docs, target, dict(s, deployment=dep), cp, b, s["parent_sha256"]
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        result.update(
            weights=digest(gen / "adapters.safetensors"),
            updates=st["step"],
            targets=sum(h["targets"] for h in st["history"]),
            update_seconds=sum(h["seconds"] for h in st["history"]),
        )
        del model, t
        gc.collect()
        mx.clear_cache()
        rationale_transfer.evaluate_pair(
            root, root / "model", panel, s, result["weights"], b, result, retention=True
        )
        f, z, r = result["full"], result["prefix_256"], result["retention"]
        result["passed"] = (
            f["all"]["wins"] - f["all"]["losses"] >= 4
            and all(
                f[k]["wins"] > f[k]["losses"] and z[k]["wins"] >= z[k]["losses"]
                for k in s["sources"]
            )
            and f["all"]["candidate"]["seconds"] <= f["all"]["base"]["seconds"]
            and r["candidate"]["nll"] <= r["base"]["nll"] + 0.03
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
