"""Train recurrent completion on untouched Base reasoning and evaluate the deployed decoder."""

import argparse
import gc
import json
import re
from pathlib import Path

import mlx.core as mx

from .. import REVISION
from ..deliberation import greedy
from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..io import checkpoint_location, digest, read, save
from . import all_layer, bootstrap, core, paired_solutions, rationale_transfer

REG = Path("research/experiments/recurrent-deliberation-01.json")
PANEL = Path("research/data/recurrent-deliberation-01.json")
TRAIN = PANEL.with_name("recurrent-deliberation-training.json")


def material(s):
    prior = [
        item
        for path in (
            paired_solutions.PANEL,
            paired_solutions.POOL,
            Path("research/data/shared-budget-vote-01.json"),
            all_layer.PANEL,
        )
        for item in read(path)
    ]
    ids = read(all_layer.TRAIN)["ordered_questions"]
    return paired_solutions.material(s, prior, exclude_ids=ids)


def decode(m, t, ids, cap, guard, *, thinking):
    if not thinking:
        raise ValueError("This evaluation requires a thinking turn")
    v = greedy(m, t, ids, cap, before=guard, stop_ids=(core.atom(t, "<|im_end|>"),))
    out = v["token_ids"]
    end = core.atom(t, "<|im_end|>")
    opening, closing = (core.atom(t, k) for k in ("<think>", "</think>"))
    boundary = out[-1] in t.eos_token_ids or out[-1] == end
    close_at = out.index(closing) if closing in out else None
    well_formed = (
        close_at is not None and out.count(closing) == 1 and opening not in out
    )
    content = out[:-1] if boundary else out
    return {
        "prompt_ids": ids,
        "token_ids": out,
        "text": t.decode(out),
        "visible": t.decode(content[close_at + 1 :]) if well_formed else "",
        "stop": ("chat_end" if out[-1] == end else "eos") if boundary else "cap",
        "close_at": close_at,
        "complete": boundary and well_formed,
        "ttft": v["ttft_s"],
        "seconds": v["elapsed_s"],
        "recurrent_from": v["recurrent_from"],
    }


def training_docs(s, pool, budget, root):
    source = Path("data/research_external/gsm-train.jsonl")
    if digest(source) != s["sources"]["gsm8k_reserved"]["sha256"]:
        raise ValueError("Wrong publisher training source")
    publisher = [json.loads(x) for x in source.read_bytes().splitlines()]
    m, t, _ = base()
    drafts, docs = [], []
    for row in pool:
        q = publisher[row["source_id"]]
        if " ".join(q["question"].split()) != row["question"]:
            raise ValueError("Wrong worked solution")
        v = core.generate(
            m,
            t,
            row["prompt_ids"],
            s["training_draft_cap"],
            budget.check,
            thinking=True,
        )
        end = v["close_at"]
        kept = len(v["token_ids"]) if end is None else end // 2
        if v["stop"] != "cap" and end is None:
            kept = (len(v["token_ids"]) - 1) // 2
        prefix = row["prompt_ids"] + v["token_ids"][:kept]
        explanation = re.sub(
            r"<<[^<>]+>>", "", q["answer"].rsplit("####", 1)[0]
        ).strip()
        response = (
            "\nLet me check. "
            + explanation
            + "\n</think>\n\nFinal answer: "
            + row["gold"]
        )
        doc = {
            "ids": prefix
            + t.encode(response, add_special_tokens=False)
            + [core.atom(t, "<|im_end|>")],
            "prefix": len(prefix),
        }
        if len(doc["ids"]) - 1 > s["cap_positions"]:
            raise ValueError("Draft supervision exceeds registered cap")
        docs.append(doc)
        drafts.append(dict(**v, id=row["id"], kept=kept, document=doc))
        packed(root / "drafts.json.gz", drafts)
    schedule = []
    for i, d in enumerate(docs):
        schedule.extend((d, docs[(i + len(docs) // 2 + 1) % len(docs)]))
    return schedule


def run(freeze=False):
    s = read(REG)
    pool, rows = material(s)
    for path, data in ((PANEL, rows), (TRAIN, pool)):
        if freeze and not path.exists():
            save(path, data)
        if read(path) != data:
            raise ValueError("Frozen deliberation material changed")
    if freeze:
        return
    compare(s, REG, PANEL, TRAIN, rows, lambda b, root: training_docs(s, pool, b, root))


def compare(s, registration, panel, training, rows, build_docs):
    index = read("research/registry.json")
    if index["experiments"][s["id"]]["registration"] != digest(registration) or any(
        index["datasets"][str(p)]["sha256"] != digest(p) for p in (panel, training)
    ):
        raise ValueError("Unregistered deliberation study")
    for key, h in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(s[key]) / "adapters.safetensors") != s[h]:
            raise ValueError("Parent or dense control changed")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    b = core.Budget(s["id"], s["limits"])
    result = {
        "registration": digest(registration),
        "data": digest(panel),
        "training": digest(training),
        "passed": False,
    }
    try:
        docs = build_docs(b, root)
        gc.collect()
        mx.clear_cache()
        m, t, state = load_trained(s["parent"])
        dep = state["spec"]["deployment"]
        m.checkpointing = True
        cp = Path("checkpoints/research") / s["id"]
        st = core.train(m, docs, dict(s, deployment=dep), cp, b, s["parent_sha256"])
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        w = digest(gen / "adapters.safetensors")
        save(
            root / "model/deliberation.json",
            {
                "revision": REVISION,
                "version": 1,
                "switch_after": s["switch_after"],
                "adapter_sha256": w,
            },
        )
        result.update(
            weights=w,
            updates=st["step"],
            targets=sum(h["targets"] for h in st["history"]),
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        rationale_transfer.evaluate_pair(
            root, root / "model", rows, s, w, b, result, retention=True, decoder=decode
        )
        full, short, r = result["full"], result["prefix_256"], result["retention"]
        result["passed"] = (
            full["all"]["wins"] - full["all"]["losses"] >= s.get("net_floor", 4)
            and all(
                full[k]["wins"] > full[k]["losses"]
                and short[k]["wins"] >= short[k]["losses"]
                for k in s["sources"]
            )
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
            and r["candidate"]["nll"] <= r["base"]["nll"] + 0.03
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
