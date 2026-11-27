"""Measure argument binding across fresh human sentence pairs, tables and live package inventories."""

import argparse
import gc
import random
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import qwen_tools
from ..csv_tools import tools
from ..evaluation.mechanism import packed, unpack
from ..interpolated_instruction import InterpolatedAssistant
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from .core import Budget
from .instruction_interpolation import funcs as file_tools
from .instruction_interpolation import plain, referent, write_documents
from .instruction_interpolation import score as file_score
from .native_function_data import token_pair
from .native_function_study import evaluate, executions, optimize, replay, summarize
from .observation_review_study import paired
from .reading_pairs import MatchedExamples
from .registered import registered


def material(spec):
    panel = read(spec["material"])
    retention = read(spec["retention_material"])
    assert digest(Path(spec["retention_material"])) == spec["retention_sha256"]
    return {
        "documents": {**panel["documents"], **retention["documents"]},
        "rows": [
            *panel["rows"],
            *[w for w in retention["rows"] if w["family"] == "inventory"],
        ],
    }


def functions(row, folder):
    if row["family"] == "inventory":
        return file_tools(row, folder)
    return (
        tools([Path(folder) / row["document"]], keyed=True)
        if row["family"] == "table"
        else []
    )


def grade(w, i, tr):
    if w["family"] == "inventory":
        return file_score(w, i, tr)
    done = tr["status"] == "answered"
    answer = tr["answer"] or ""
    if w["family"] == "meaning":
        ok = referent(answer, w["options"]) == plain(w["gold"])
        return {"completed": done, "correct_answer": ok, "success": done and ok}
    gold = plain(w["gold"])
    text = plain(answer.strip('"'))
    contains = bool(re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", text))
    calls = executions(tr)
    observed = any(
        e
        == {
            "call": {"name": "csv_lookup", "arguments": w["arguments"]},
            "observation": {"value": w["gold"]},
        }
        for e in calls
    )
    return {
        "completed": done,
        "correct_answer": contains,
        "exact": text == gold,
        "observed": observed,
        "success": done and contains and observed,
    }


def outcome(data, runs):
    metrics = {}
    solved = {}
    strict = {}
    for side, arm in runs.items():
        grouped = defaultdict(list)
        for w, r in zip(data["rows"], arm, strict=True):
            assert w["id"] == r["id"] and w["family"] == r["family"]
            grouped[w["family"], w["cluster"]].extend(
                q["score"]["success"] for q in r["requests"]
            )
        assert all(len(v) == 2 for v in grouped.values())
        solved[side] = {k for k, v in grouped.items() if all(v)}
        m = metrics[side] = {
            f: summarize([r for r in arm if f == "all" or r["family"] == f])
            for f in ("all", "meaning", "table", "inventory")
        }
        q = [q for r in arm for q in r["requests"]]
        turns = [t for z in q for t in z["trajectory"]["turns"]]
        m["opening_rate"] = (
            sum(len(z["trajectory"]["turns"][0]["prompt_ids"]) for z in q)
            / m["all"]["first_s"]
        )
        m["decode_rate"] = sum(len(t["token_ids"]) - 1 for t in turns) / sum(
            t["seconds"] - t["ttft_s"] for t in turns
        )
        strict[side] = sum(
            q["score"]["exact"] and q["score"]["observed"]
            for r in arm
            if r["family"] == "table"
            for q in r["requests"]
        )
    pairs = {}
    for f in ("meaning", "table", "inventory", "primary"):
        a, b = [
            {
                k
                for k in solved[s]
                if (k[0] != "inventory" if f == "primary" else k[0] == f)
            }
            for s in ("candidate", "control")
        ]
        pairs[f] = {"loop": len(a), "base": len(b), **paired(a, b)}
    a, b = metrics["candidate"], metrics["control"]
    gates = {
        "reasoning": pairs["meaning"]["wins"] > pairs["meaning"]["losses"],
        "tables": pairs["table"]["wins"] > pairs["table"]["losses"]
        and strict["candidate"] >= strict["control"],
        "primary": pairs["primary"]["wins"] - pairs["primary"]["losses"] >= 5
        and pairs["primary"]["p"] <= 0.05,
        "field": pairs["inventory"]["loop"] >= pairs["inventory"]["base"]
        and a["inventory"]["grounded"] >= max(2, b["inventory"]["grounded"]),
        "retention": all(
            a[f][k] >= b[f][k]
            for f in ("meaning", "table", "inventory")
            for k in ("completed", "answers")
        ),
        "wall": a["all"]["wall_s"] < b["all"]["wall_s"],
        "first": a["all"]["first_s"] < b["all"]["first_s"],
        "peak": a["all"]["peak"] < b["all"]["peak"],
        "rates": all(a[k] > b[k] for k in ("opening_rate", "decode_rate")),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "pairs": pairs,
        "groups": metrics,
        "exact_grounded_table_responses": strict,
    }


def run(name):
    spec = registered(name)
    data = material(spec)
    release = Path(spec["requires"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    assert all(
        digest(Path(spec["checkpoint"]) / p) == v
        for p, v in read(release / "export-files.json").items()
    )
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            optimize(spec, root, budget)
        with TemporaryDirectory() as tmp:
            write_documents(data, tmp)
            runs = {}

            def control():
                model = DenseAssistant(True, "qwen", spec["checkpoint"])
                model.native_plain = True
                return model

            for side in ("candidate", "control"):
                gc.collect()
                mx.clear_cache()
                dest = root / (side + ".json.gz")
                if not dest.exists():
                    actor = (
                        (lambda: InterpolatedAssistant(spec["checkpoint"], root))
                        if side == "candidate"
                        else control
                    )
                    packed(
                        dest,
                        evaluate(
                            spec,
                            data,
                            side,
                            budget,
                            actor,
                            lambda w: functions(w, tmp),
                            scorer=grade,
                        ),
                    )
                runs[side] = unpack(dest)
            save(root / "decision.json", outcome(data, runs))
    finally:
        budget.close()


def verify(root, d):
    spec = registered(read(root / "registration.json")["id"])
    assert read(root / "registration.json") == spec
    fit = read(root / "fit.json")
    train = read(spec["training"])["examples"]
    assert (
        fit["trainable"] == 802816
        and fit["passes"] == 2
        and fit["turns"] == len(train)
        and fit["updates"] == spec["updates"]
    )
    assert fit["weights"] == digest(root / "tool-adapters.safetensors") and fit[
        "training"
    ] == digest(Path(spec["training"]))
    data = material(spec)
    tok = load_tokenizer(local_snapshot())
    bank = [(None, 0, token_pair(tok, r)[1]) for r in train]
    pairs = MatchedExamples(train, bank, tok, spec["pairs_per_batch"], spec["batch"])
    rng = random.Random(spec["fit_seed"])
    assert fit["targets"] == sum(
        sum(len(q[2]) for q in pairs.sample(rng)) for _ in range(spec["updates"])
    )
    runs = {}
    with TemporaryDirectory() as tmp:
        write_documents(data, tmp)
        for side in ("candidate", "control"):
            arm = runs[side] = unpack(root / (side + ".json.gz"))
            for w, r in zip(data["rows"], arm, strict=True):
                fs = functions(w, tmp)
                history = []
                assert (
                    w["initial_ids"]
                    == r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                )
                assert w["initial_ids"] == tok.encode(
                    qwen_tools.render(
                        tok,
                        [{"role": "user", "content": w["questions"][0]}],
                        qwen_tools.declarations(fs),
                    ),
                    add_special_tokens=False,
                )
                for i, (question, q) in enumerate(
                    zip(w["questions"], r["requests"], strict=True)
                ):
                    history.append({"role": "user", "content": question})
                    replay(tok, w, fs, history, q, spec, side)
                    assert grade(w, i, q["trajectory"]) == q["score"]
    assert outcome(data, runs) == d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study")
    run(p.parse_args().study)
