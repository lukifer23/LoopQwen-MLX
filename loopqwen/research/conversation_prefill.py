"""Measure private KV reuse on complete, unchanged documents and actual file callbacks."""

import argparse
import gc
import hashlib
import subprocess
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import file_tools
from ..evaluation import program_questions
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..tool_model import Assistant
from .core import Budget
from .native_function_study import executions
from .registered import registered


def check_tokens(item, ids):
    path = "research/data/conversation-document-prefill-tokens.json"
    assert digest(path) == read("research/registry.json")["datasets"][path]["sha256"]
    expected = read(path)[item["source"]]
    assert (
        len(ids) == expected["prompt_tokens"]
        and fingerprint(ids) == expected["initial_sha256"]
    )


def material(spec, item):
    data = subprocess.check_output(
        ["git", "show", spec["revision"] + ":" + item["source"]]
    )
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    file = Path(item["file"])
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        file.write_bytes(data)
    assert file.read_bytes() == data
    text = data.decode()
    assert item["evidence"] in text
    questions = [
        f"Here is the full text of {file}.\n\n{text}\n\n{item['question']}",
        f"Now check the actual file {file} with the available tool. How many bytes is it?",
        "And how many lines does that same file contain? Please check with the file tool.",
    ]
    return file, questions, [item["answer"], len(data), len(data.splitlines())]


def grade(tr, index, path, value):
    calls = executions(tr)
    correct = program_questions.score({"gold": str(value)}, {**tr, "turns": []})[
        "correct_answer"
    ]
    if index:
        key = ["bytes", "lines"][index - 1]
        grounded = any(
            a["call"] == {"name": "file_" + key, "arguments": {"path": str(path)}}
            and a["observation"] == {key: value}
            for a in calls
        )
    else:
        grounded = True
    return {
        "answer": correct,
        "success": correct and grounded and tr["status"] == "answered",
    }


def timed(spec, item, side, budget):
    file, questions, answers = material(spec, item)
    available = file_tools.tools([file])
    mx.clear_cache()
    mx.reset_peak_memory()
    began = perf_counter()
    assistant = Assistant(spec["checkpoint"], tool_format=spec["format"])
    chat = assistant.conversation(available, retain_prefix=side == "retain")
    rows = []
    for i, (question, answer) in enumerate(zip(questions, answers, strict=True)):
        stamps = {}
        start = began if not i else perf_counter()

        def observe(n, times=stamps, origin=start):
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
        if i == 0:
            initial = tr["turns"][0]["prompt_ids"]
            check_tokens(item, initial)
        rows.append(
            {
                "trajectory": tr,
                "first": stamps[len(tr["turns"][0]["prompt_ids"])],
                "wall": perf_counter() - start,
                "peak": mx.get_peak_memory(),
                "score": grade(tr, i, file, answer),
            }
        )
    return rows


def outcome(root, spec):
    from scripts.verify_runtime_wheel import trace

    metrics = {}
    for s in ("retain", "stock"):
        rows = [
            unpack(root / f"{i}-{s}.json.gz") for i in range(len(spec["documents"]))
        ]
        follow = [q for doc in rows for q in doc[1:]]
        metrics[s] = {
            "first_followup": sum(q["first"] for q in follow),
            "wall": sum(q["wall"] for doc in rows for q in doc),
            "max_peak": max(q["peak"] for doc in rows for q in doc),
            "correct": sum(q["score"]["success"] for doc in rows for q in doc),
            "all_dialogues": sum(
                all(q["score"]["success"] for q in doc) for doc in rows
            ),
            "grounded_by_document": [
                sum(q["score"]["success"] for q in doc[1:]) for doc in rows
            ],
            "first_turn_reused": [
                [
                    q["trajectory"]["turns"][0].get("reused_prompt_tokens", 0)
                    for q in doc
                ]
                for doc in rows
            ],
        }
    exact = []
    for i, item in enumerate(spec["documents"]):
        file, _, answers = material(spec, item)
        a, b = [unpack(root / f"{i}-{s}.json.gz") for s in ("retain", "stock")]
        assert len(a) == len(b) == len(answers)
        for j, (x, y) in enumerate(zip(a, b, strict=True)):
            for v in (x, y):
                assert grade(v["trajectory"], j, file, answers[j]) == v["score"]
                assert 0 < v["first"] <= v["wall"] and v["peak"] > 0
            exact.append(trace(x["trajectory"]) == trace(y["trajectory"]))
    a, b = metrics["retain"], metrics["stock"]
    gates = {
        "exact": all(exact),
        "grounded": all(a["grounded_by_document"]),
        "reused": all(v[0] == 0 and min(v[1:]) > 4096 for v in a["first_turn_reused"]),
        "stock": all(not any(v) for v in b["first_turn_reused"]),
        "latency": a["first_followup"] < 0.7 * b["first_followup"],
        "wall": a["wall"] < b["wall"],
        "peak": a["max_peak"] <= 1.25 * b["max_peak"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "identical": exact,
    }


def verify(root, d):
    from mlx_lm.utils import load_tokenizer

    from ..model_loader import local_snapshot
    from ..tool_session import system_message
    from .mixed_field import replay_exchange

    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    assert d == outcome(root, spec)
    tok = load_tokenizer(local_snapshot())
    for i, item in enumerate(spec["documents"]):
        file, questions, _ = material(spec, item)
        available = file_tools.tools([file])
        for side in ("retain", "stock"):
            rows = unpack(root / f"{i}-{side}.json.gz")
            initial = rows[0]["trajectory"]["turns"][0]["prompt_ids"]
            check_tokens(item, initial)
            history = [system_message(available)]
            for question, row in zip(questions, rows, strict=True):
                history.append({"role": "user", "content": question})
                tr = row["trajectory"]
                replay_exchange(tok, available, history, tr, spec)
                for turn in tr["turns"]:
                    length = len(turn["prompt_ids"]) + len(turn["token_ids"]) - 1
                    sites = turn["kv_history"]
                    assert {(s["layer"], s["turn"]) for s in sites} == {
                        *((layer, 0) for layer in range(3, 24, 4)),
                        (23, 1),
                    }
                    assert all(s["offset"] == s["retained"] == length for s in sites)


def run(name):
    spec = registered(name)
    parent = Path(spec["requires"])
    assert (
        read(parent / "decision.json")["passed"]
        and digest(parent / "checksums.json") == spec["prerequisite_sha256"]
    )
    assert all(
        digest(Path(spec["checkpoint"]) / k) == v
        for k, v in read(parent / "export-files.json").items()
    )
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        for i, item in enumerate(spec["documents"]):
            for side in spec["order"][i]:
                dest = root / f"{i}-{side}.json.gz"
                if not dest.exists():
                    gc.collect()
                    packed(dest, timed(spec, item, side, budget))
                    print(i, side, flush=True)
        save(root / "decision.json", outcome(root, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
