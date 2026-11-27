"""Compare a qualified loop and untouched Base on the same equality-checking calculator."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..calculator_tool import tool
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import read, save, digest
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_model import Assistant
from ..tool_session import initial_messages
from .core import Budget
from .native_function_study import evaluate, grade, summarize, executions
from .mixed_field import replay_exchange
from .registered import registered


def catalog(_):
    return [tool(check_equalities=True)]


def outcome(data):
    stats = {
        s: {
            group: summarize(
                [r for r in rows if group == "all" or group == r["family"]]
            )
            for group in ("all", "ASDiv", "SVAMP")
        }
        for s, rows in data.items()
    }
    changes = {}
    for key in ("correct_answer", "success"):
        dif = [
            int(a["requests"][0]["score"][key]) - int(b["requests"][0]["score"][key])
            for a, b in zip(data["candidate"], data["control"], strict=True)
        ]
        gains, losses = dif.count(1), dif.count(-1)
        changes[key] = dict(gains=gains, losses=losses, p=exact_mcnemar(gains, losses))
    verified = {}
    for side, rows in data.items():
        counts = []
        for r in rows:
            tr = r["requests"][0]
            calls = executions(tr["trajectory"])
            checked = any(
                "value" in c["observation"]
                and any("=" in step for step in c["call"]["arguments"]["steps"])
                for c in calls
            )
            counts.append((checked, tr["score"]["success"] and checked))
        verified[side] = dict(
            executed=sum(x for x, y in counts), correct=sum(y for x, y in counts)
        )
    a, b = stats["candidate"], stats["control"]
    x, y = a["all"], b["all"]
    gates = dict(
        answer=x["answers"] >= y["answers"] + 3
        and changes["correct_answer"]["p"] <= 0.05,
        grounded=x["strict"] >= y["strict"] + 4 and changes["success"]["p"] <= 0.05,
        domains=all(a[k]["answers"] >= b[k]["answers"] for k in ("ASDiv", "SVAMP")),
        real_equalities=verified["candidate"]["correct"] >= 3,
        costs=all(x[k] <= y[k] for k in ("wall_s", "first_s", "peak")),
    )
    return dict(
        passed=all(gates.values()),
        gates=gates,
        groups=stats,
        changes=changes,
        equalities=verified,
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    panel = read(spec["material"])
    tok = load_tokenizer(local_snapshot())
    data = {}
    for side in ("candidate", "control"):
        rows = unpack(root / (side + ".json.gz"))
        data[side] = rows
        for w, r in zip(panel["rows"], rows, strict=True):
            assert (r["id"], r["family"]) == (w["id"], w["family"]) and len(
                r["requests"]
            ) == 1
            q = r["requests"][0]
            tr = q["trajectory"]
            tools = catalog(w)
            assert tr["turns"][0]["prompt_ids"] == w["initial_ids"] and q[
                "score"
            ] == grade(w, 0, tr)
            assert (
                0 < q["first_s"] <= q["wall_s"]
                and 0 < tr["seconds"] <= q["wall_s"]
                and q["peak"] > 0
            )
            replay_exchange(
                tok, tools, initial_messages(w["questions"][0], tools), tr, spec
            )
            for t in tr["turns"]:
                sites = {(v["layer"], v["turn"]) for v in t["kv_history"]}
                assert sites == {(i, 0) for i in (3, 7, 11, 15, 19, 23)} | (
                    {(23, 1)} if side == "candidate" else set()
                )
                assert all(
                    v["offset"]
                    == v["retained"]
                    == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                    for v in t["kv_history"]
                )
    assert outcome(data) == decision


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    parent = Path(spec["requires"])
    assert (
        read(parent / "decision.json")["passed"]
        and digest(parent / "checksums.json") == spec["prerequisite_sha256"]
    )
    files = read(parent / "export-files.json")
    checkpoint = Path(spec["checkpoint"])
    assert read(checkpoint / "tool-export-files.json") == files and all(
        digest(checkpoint / k) == v for k, v in files.items()
    )
    panel = read(spec["material"])
    data = {}
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        for side in ("candidate", "control"):
            gc.collect()
            mx.clear_cache()
            dest = root / (side + ".json.gz")
            if not dest.exists():
                factory = (
                    (lambda: Assistant(checkpoint))
                    if side == "candidate"
                    else (lambda: DenseAssistant(True))
                )
                packed(dest, evaluate(spec, panel, side, budget, factory, catalog))
            data[side] = unpack(dest)
        save(root / "decision.json", outcome(data))
    finally:
        budget.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("name")
    run(a.parse_args().name)
