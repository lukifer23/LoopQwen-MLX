"""Read complete third-party books, then operate on the same files in native tool dialogue."""

import argparse
import csv
import gc
import re
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import file_tools
from ..evaluation.mechanism import packed, unpack
from ..evaluation.service_answers import visible_reference
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_model import Assistant
from .core import Budget
from .native_function_study import executions, replay
from .registered import registered


def grade(item, index, tr):
    completed = tr["status"] == "answered"
    value = [item["reference"], item["size"], item["lines"]][index]
    text = tr["answer"] or ""
    if not index:
        for word, number in (("twenty-five", 25), ("twenty five", 25), ("thirty", 30)):
            text = re.sub(r"\b" + word + r"\b", str(number), text, flags=re.IGNORECASE)
    meaning = visible_reference(text, item["file"], str(value), file=bool(index))
    events = executions(tr)
    if not index:
        correct = []
    else:
        key = ("bytes", "lines")[index - 1]
        action = {"name": "file_" + key, "arguments": {"path": item["file"]}}
        correct = [
            e["call"] == action and e["observation"] == {key: value} for e in events
        ]
    path = not index or any(correct)
    return {
        "success": completed and meaning and path,
        "meaning": completed and meaning,
        "exact_path": not index or (len(correct) == 1 and all(correct)),
        "calls": len(events),
    }


def measure(spec, item, side, budget):
    from scripts.book_conversation_material import messages, source

    path = source(item)
    assert (
        path.stat().st_size == item["size"]
        and len(path.read_bytes().splitlines()) == item["lines"]
    )
    questions = messages(item)
    available = file_tools.tools([path])
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    actor = (
        Assistant(spec["checkpoint"], tool_format="qwen")
        if side == "candidate"
        else DenseAssistant(True, "qwen")
    )
    chat = actor.conversation(available, retain_prefix=True)
    rows = []
    for i, question in enumerate(questions):
        stamps = {}
        begin = start if i == 0 else perf_counter()

        def observe(n, times=stamps, origin=begin):
            times.setdefault(n, perf_counter() - origin)
            budget.check()

        tr = chat.respond(
            question,
            max_tokens=spec["max_tokens"],
            max_turns=spec["max_turns"],
            reuse_prefix=True,
            record_kv_history=True,
            progress=observe,
        )
        mx.synchronize()
        wall = perf_counter() - begin
        first = tr["turns"][0]["prompt_ids"]
        if not i:
            assert first == item["initial_ids"]
        rows.append(
            {
                "trajectory": tr,
                "wall_s": wall,
                "first_s": stamps[len(first)],
                "peak": mx.get_peak_memory(),
                "score": grade(item, i, tr),
            }
        )
        print(side, item["title"], i + 1, round(wall, 2), flush=True)
    return rows


def outcome(root, panel):
    metrics = {}
    for side in ("candidate", "control"):
        docs = [unpack(root / f"{i}-{side}.json.gz") for i in range(len(panel))]
        rows = [q for doc in docs for q in doc]
        turns = [t for q in rows for t in q["trajectory"]["turns"]]
        tokens = sum(len(t["token_ids"]) - 1 for t in turns)
        decode = sum(t["seconds"] - t["ttft_s"] for t in turns)
        m = {
            "n": len(rows),
            "success": sum(q["score"]["success"] for q in rows),
            "books": sum(doc[0]["score"]["success"] for doc in docs),
            "metadata": sum(q["score"]["success"] for doc in docs for q in doc[1:]),
            "whole": sum(all(q["score"]["success"] for q in doc) for doc in docs),
            "wall": sum(q["wall_s"] for q in rows),
            "first": sum(q["first_s"] for q in rows),
            "peak": max(q["peak"] for q in rows),
            "initial_prefill_tps": sum(len(item["initial_ids"]) for item in panel)
            / sum(doc[0]["first_s"] for doc in docs),
            "decode_tps": tokens / decode,
            "output_tokens": sum(len(t["token_ids"]) for t in turns),
            "reused": [
                [q["trajectory"]["turns"][0]["reused_prompt_tokens"] for q in doc]
                for doc in docs
            ],
        }
        metrics[side] = m
    a, b = metrics["candidate"], metrics["control"]
    gates = {
        "helpful": a["success"] == a["n"] and a["success"] > b["success"],
        "history": all(
            v[0] == 0 and min(v[1:]) >= len(item["initial_ids"]) - 256
            for model in metrics.values()
            for v, item in zip(model["reused"], panel, strict=True)
        ),
        **{k: a[k] < b[k] for k in ("wall", "first", "peak")},
        **{k: a[k] > b[k] for k in ("initial_prefill_tps", "decode_tps")},
    }
    return {"passed": all(gates.values()), "gates": gates, "metrics": metrics}


def verify(root, d):
    from scripts.book_conversation_material import PUBLIC, messages, source

    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    material = read(spec["material"])
    assert (
        digest(PUBLIC / "qaps.csv") == material["annotation_sha256"]
        and digest(PUBLIC / "documents.csv") == material["documents_sha256"]
    )
    qas = list(csv.DictReader((PUBLIC / "qaps.csv").open()))
    tok = load_tokenizer(local_snapshot())
    for i, item in enumerate(material["items"]):
        q = next(
            q
            for q in qas
            if q["document_id"] == item["document_id"]
            and q["question"] == item["question"]
        )
        assert [q["answer1"], q["answer2"]] == item["annotations"]
        path = source(item)
        assert (
            path.stat().st_size == item["size"]
            and len(path.read_bytes().splitlines()) == item["lines"]
        )
        available = file_tools.tools([path])
        for side in ("candidate", "control"):
            history = []
            rows = unpack(root / f"{i}-{side}.json.gz")
            assert (
                rows[0]["trajectory"]["turns"][0]["prompt_ids"] == item["initial_ids"]
            )
            for j, (question, row) in enumerate(zip(messages(item), rows, strict=True)):
                history.append({"role": "user", "content": question})
                replay(tok, None, available, history, row, spec, side)
                assert row["score"] == grade(item, j, row["trajectory"])
    assert outcome(root, material["items"]) == d


def run(name):
    spec = registered(name)
    parent = Path(spec["requires"])
    assert (
        digest(parent / "checksums.json") == spec["prerequisite_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    assert all(
        digest(Path(spec["checkpoint"]) / k) == v
        for k, v in read(Path(spec["release"]) / "export-files.json").items()
    )
    panel = read(spec["material"])["items"]
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        for i, order in enumerate(spec["order"]):
            for side in order:
                dest = root / f"{i}-{side}.json.gz"
                if not dest.exists():
                    packed(dest, measure(spec, panel[i], side, budget))
        save(root / "decision.json", outcome(root, panel))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
