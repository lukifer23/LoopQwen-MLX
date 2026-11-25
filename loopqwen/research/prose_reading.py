"""Evaluate the frozen attention reader on natural extractive questions."""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION
from ..deployment import load_trained
from ..evaluation import extractive
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import core

REG = Path("research/experiments/prose-reading-01.json")
DATA = Path("research/data/prose-reading-01.json")


def material(t, s):
    source = s["source"]
    path = Path("data/research_external/squad-dev-v1.1.json")
    if not path.exists():
        with urlopen(source["url"], timeout=45) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Publisher data changed")
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    if digest(path) != source["sha256"]:
        raise ValueError("Wrong SQuAD release")
    articles = {a["title"]: a["paragraphs"] for a in read(path)["data"]}
    panel = []
    for i, title in enumerate(s["titles"]):
        rng = random.Random(s["seed"] + i)
        pars = articles[title]
        index, qa = rng.choice([(j, q) for j, p in enumerate(pars) for q in p["qas"]])
        distractors = [j for j in range(len(pars)) if j != index]
        rng.shuffle(distractors)
        group = []
        for other in distractors:
            group.append(other)
            pos = int(len(group) * (0.1, 0.5, 0.9)[i % 3])
            order = group[:pos] + [index] + group[pos:]
            body = "\n\n".join(pars[j]["context"] for j in order)
            user = (
                body
                + "\n\nQuestion: "
                + qa["question"]
                + "\nReason briefly. Give just the shortest phrase from the article that answers the question."
            )
            text = t.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            ids = t.encode(text, add_special_tokens=False)
            if len(ids) >= s["minimum_length"]:
                break
        if len(ids) < s["minimum_length"] or t.decode(ids) != text:
            raise ValueError("Natural reading context length or tokenizer roundtrip")
        start = text.index(pars[index]["context"])
        lower = len(t.encode(text[:start], add_special_tokens=False))
        upper = len(
            t.encode(
                text[: start + len(pars[index]["context"])], add_special_tokens=False
            )
        )
        panel.append(
            {
                "id": qa["id"],
                "title": title,
                "question": qa["question"],
                "aliases": sorted({a["text"] for a in qa["answers"]}),
                "placement": ("early", "middle", "late")[i % 3],
                "source_paragraph": index,
                "order": order,
                "source_interval": [lower, upper],
                "prompt_ids": ids,
            }
        )
    return panel


def run(freeze):
    s = read(REG)
    parent = Path(s["qualification"])
    if (
        digest(parent / "checksums.json") != s["qualification_sha256"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Qualification changed")
    t = load_tokenizer(local_snapshot())
    items = material(t, s)
    if freeze and not DATA.exists():
        DATA.write_text(json.dumps(items, separators=(",", ":")) + "\n")
    if read(DATA) != items:
        raise ValueError("Frozen public reading cohort differs")
    if freeze:
        return
    if read("research/registry.json")["datasets"][str(DATA)]["sha256"] != digest(DATA):
        raise ValueError("Unregistered natural questions")
    root = core.OUT / s["id"]
    if root.exists():
        raise FileExistsError(root)
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    rows = {"base": [], "candidate": []}
    decision = {
        "passed": False,
        "data": digest(DATA),
        "registration": digest(REG),
        "qualification": s["qualification_sha256"],
    }
    try:
        half = len(items) // 2
        for start, order in ((0, ("base", "candidate")), (half, ("candidate", "base"))):
            for arm in order:
                m, t, _ = base() if arm == "base" else load_trained(parent / "model")
                for item in items[start : start + half]:
                    ids = item["prompt_ids"]
                    v = generate(
                        m,
                        t,
                        t.decode(ids),
                        s["generation_cap"],
                        progress=lambda _: b.check(),
                        stop_ids=[core.atom(t, "<|im_end|>")],
                    )
                    del v["prompt"]
                    ans = extractive.answer(v["token_ids"], v["text"])
                    em, f1 = extractive.score(ans, item["aliases"])
                    v.update(
                        id=item["id"],
                        input_sha256=fingerprint(ids),
                        answer=ans,
                        correct=em,
                        f1=f1,
                    )
                    if arm == "candidate":
                        lo, hi = item["source_interval"]
                        sec = v["attention_memory"]
                        v["selected_source"] = lo < sec["stop"] and hi > sec["start"]
                    rows[arm].append(v)
                    packed(
                        root / (arm + ".json.gz"),
                        {
                            "weights": REVISION
                            if arm == "base"
                            else s["qualification_sha256"],
                            "records": rows[arm],
                        },
                    )
                del m, t
                gc.collect()
                mx.clear_cache()
        scores = {}
        for k, rs in rows.items():
            scores[k] = {
                "exact": sum(r["correct"] for r in rs),
                "f1": sum(r["f1"] for r in rs) / len(rs),
                "seconds": sum(r["elapsed_s"] for r in rs),
                "completed": sum(r["answer"] is not None for r in rs),
                "placement": {
                    pos: sum(
                        r["correct"]
                        for r, item in zip(rs, items)
                        if item["placement"] == pos
                    )
                    for pos in ("early", "middle", "late")
                },
            }
        a, c = scores["base"], scores["candidate"]
        decision.update(
            status="completed",
            scores=scores,
            selected_source=sum(r["selected_source"] for r in rows["candidate"]),
            passed=c["exact"] >= 8
            and c["exact"] - a["exact"] >= 2
            and c["f1"] >= a["f1"]
            and c["seconds"] <= 1.5 * a["seconds"],
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        decision.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    run(parser.parse_args().freeze)
