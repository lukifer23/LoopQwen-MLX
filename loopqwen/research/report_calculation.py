"""Train and score freely interleaved arithmetic on actual reports and held-out tables."""

import argparse
import ast
import gc
import re
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..calculator_tool import calculate, tool
from ..evaluation.answers import NUMBER, numeric
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_session import compact, initial_messages, prompt
from ..tool_session import run as session
from .core import Budget
from .mixed_field import replay_exchange
from .registered import registered
from .tool_fit import fitted, restore


def number(text):
    value = numeric(text.strip().rstrip("%"))
    if value is not None:
        return Fraction(value), True
    matches = re.findall(rf"({NUMBER}(?:\s*/\s*{NUMBER})?)", text)
    tail = numeric(matches[-1]) if matches else None
    return (Fraction(tail), False) if tail is not None else (None, False)


def operation_evidence(expression, prefix, previous):
    linked = substantive = False
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if not isinstance(node, ast.BinOp):
            continue
        values = [
            Fraction(calculate([*prefix, ast.unparse(part)])["value"])
            for part in (node.left, node.right)
        ]
        left, right = values
        linked |= any(v in previous for v in values)
        if isinstance(node.op, ast.Add):
            substantive |= left != 0 and right != 0
        elif isinstance(node.op, ast.Sub):
            substantive |= right != 0
        elif isinstance(node.op, ast.Mult):
            substantive |= left not in (0, 1) and right not in (0, 1)
        elif isinstance(node.op, ast.Div):
            substantive |= right != 1
    return substantive, linked


def grade(w, tr):
    gold, tol = Fraction(w["gold"]), Fraction(w["tolerance"])
    answer, strict = number(tr["answer"] or "")
    done = tr["status"] == "answered"
    correct = done and answer is not None and abs(answer - gold) <= tol
    calls = [t for t in tr["turns"] if "observation" in t]
    valid = [t for t in calls if "value" in t["observation"]]
    arithmetic = False
    linked = False
    previous = set()
    for t in valid:
        steps = t["call"]["arguments"]["steps"]
        for i, expr in enumerate(steps):
            nonidentity, uses_prior = operation_evidence(expr, steps[:i], previous)
            arithmetic |= nonidentity
            linked |= uses_prior and nonidentity
        previous.add(Fraction(t["observation"]["value"]))
    last = Fraction(valid[-1]["observation"]["value"]) if valid else None
    path = (
        bool(calls)
        and len(valid) == len(calls)
        and arithmetic
        and abs(last - gold) <= tol
    )
    expert = None
    if "steps" in w:
        # Order-sensitive check on the actual intermediate values of the published route.
        expected = [Fraction(calculate([s])["value"]) for s in w["steps"]]
        got = [Fraction(v) for t in valid for v in t["observation"]["results"]]
        expert = bool(path and got == expected)
    return {
        "correct": correct,
        "strict": correct and strict,
        "grounded": correct and path,
        "completed": done,
        "calls": len(calls),
        "linked_grounded": bool(correct and path and linked),
        "expert_values": expert,
    }


def outcome(panel, arms, spec):
    groups = {}
    for source in ("all", "FinQA", "TabMWP"):
        ix = [i for i, w in enumerate(panel) if source in ("all", w["source"])]
        sides = {}
        for side, rows in arms.items():
            sub = [rows[i] for i in ix]
            sides[side] = {
                "n": len(ix),
                **{
                    k: sum(int(r["score"][k] or 0) for r in sub)
                    for k in (
                        "correct",
                        "strict",
                        "grounded",
                        "completed",
                        "linked_grounded",
                        "expert_values",
                        "calls",
                    )
                },
                "seconds": sum(r["trajectory"]["seconds"] for r in sub),
                "first": sum(r["trajectory"]["turns"][0]["ttft_s"] for r in sub),
                "peak": max(
                    t["peak_bytes"] for r in sub for t in r["trajectory"]["turns"]
                ),
            }
        diffs = [
            int(arms["candidate"][i]["score"]["correct"])
            - int(arms["control"][i]["score"]["correct"])
            for i in ix
        ]
        groups[source] = {
            **sides,
            "gains": diffs.count(1),
            "losses": diffs.count(-1),
            "p": exact_mcnemar(diffs.count(1), diffs.count(-1)),
        }
    all_ = groups["all"]
    a, b = all_["candidate"], all_["control"]
    passed = (
        all_["gains"] - all_["losses"] >= 6
        and all_["p"] <= 0.05
        and a["grounded"] >= max(16, b["grounded"] + 10)
        and a["linked_grounded"] >= 6
        and all(a[k] <= b[k] for k in ("seconds", "first", "peak"))
        and all(
            g["gains"] - g["losses"] >= 2
            and all(
                g["candidate"][k] >= g["control"][k]
                for k in ("strict", "completed", "grounded")
            )
            for g in (groups["FinQA"], groups["TabMWP"])
        )
    )
    return {"passed": passed, "groups": groups}


def verify(root, d):
    from scripts.report_calculation_material import request

    spec = read(root / "registration.json")
    mat = read(spec["material"])
    panel = mat["rows"]
    train = read(spec["training"])
    assert set(mat["held_out_issuers"]).isdisjoint(
        w["cluster"] for w in train["episodes"] if w["source"] == "FinQA"
    )
    assert all(
        len({w["cluster"] for w in panel if w["source"] == s}) == 24
        for s in ("FinQA", "TabMWP")
    )
    tok = load_tokenizer(local_snapshot())
    tools = [tool()]
    meta = read(root / "fit.json")
    assert (
        meta["passes"] == 2
        and meta["turns"] == len(train["examples"])
        and meta["updates"] == spec["updates"]
        and meta["weights"] == digest(root / "tool-adapters.safetensors")
        and meta["training"] == digest(Path(spec["training"]))
    )
    by_id = {r["id"]: r for r in train["examples"]}
    for w in train["episodes"]:
        msgs = initial_messages(request(w), tools)
        for i, step in enumerate(w["steps"]):
            r = by_id[w["id"] + f"-step-{i}"]
            args = {"steps": [step]}
            assert r["messages"] == msgs and r["visible"] == compact(
                {"name": "calculate", "arguments": args}
            )
            msgs.extend(
                [
                    {"role": "assistant", "content": r["visible"]},
                    {"role": "tool", "content": compact(calculate(**args))},
                ]
            )
        final = by_id[w["id"] + "-final"]
        assert final["messages"] == msgs and final["visible"] == w["gold"]
    arms = {}
    for side in ("candidate", "control"):
        rows = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel, rows, strict=True):
            tr = r["trajectory"]
            msgs = initial_messages(request(w), tools)
            assert (
                w["user_request"] == request(w)
                and r["id"] == w["id"]
                and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
            )
            assert (
                tok.encode(prompt(tok, msgs), add_special_tokens=False)
                == w["initial_ids"]
            )
            replay_exchange(tok, tools, msgs, tr, spec)
            for t in tr["turns"]:
                sites = {(v["layer"], v["turn"]) for v in t["kv_history"]}
                expected = {(j, 0) for j in (3, 7, 11, 15, 19, 23)}
                if side == "candidate":
                    expected.add((23, 1))
                assert sites == expected and all(
                    v["offset"]
                    == v["retained"]
                    == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                    for v in t["kv_history"]
                )
            assert r["score"] == grade(w, tr)
        arms[side] = rows
    assert outcome(panel, arms, spec) == d


def run(name):
    spec = registered(name)
    parent = Path(spec["initial_adapter"])
    if (
        not read(parent / "decision.json")["passed"]
        or digest(parent / "checksums.json") != spec["initial_sha256"]
    ):
        raise ValueError("Qualified tool parent differs")
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    budget = Budget(name, spec["limits"])
    arms = {}
    try:
        panel = read(spec["material"])["rows"]
        for side in ("candidate", "control"):
            if side == "candidate":
                if (root / "fit.json").exists():
                    # restore() verifies the weight set and registered graph.
                    save(root / "registration.json", spec)
                    model, tok = restore(root)
                else:
                    model, tok = fitted("train_loop", spec, root, budget)
            else:
                model, tok = base()[:2]
            model.eval()
            mx.clear_cache()
            mx.reset_peak_memory()
            path = root / (side + ".json.gz")
            rows = unpack(path) if path.exists() else []
            assert [r["id"] for r in rows] == [w["id"] for w in panel[: len(rows)]]
            for w in panel[len(rows) :]:
                budget.check()
                tr = session(
                    model,
                    tok,
                    w["user_request"],
                    [tool()],
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    reuse_prefix=False,
                    record_kv_history=True,
                    progress=lambda _: budget.check(),
                )
                assert tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                rows.append({"id": w["id"], "trajectory": tr, "score": grade(w, tr)})
                if len(rows) % 6 == 0 or len(rows) == len(panel):
                    packed(path, rows)
                    print(
                        side,
                        len(rows),
                        sum(r["score"]["correct"] for r in rows),
                        flush=True,
                    )
            arms[side] = rows
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(root / "decision.json", outcome(panel, arms, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
