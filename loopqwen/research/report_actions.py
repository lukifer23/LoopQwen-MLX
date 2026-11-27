"""Fit and verify a shared native layer on two different actions over the same original report."""

import argparse
import gc
import random
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import package_tools
from ..calculator_tool import tool
from ..evaluation import program_questions
from ..evaluation.mechanism import packed, unpack
from ..evaluation.service_answers import visible_reference
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_model import Assistant
from .core import Budget
from .native_function_data import catalog, token_pair
from .native_function_study import (
    evaluate,
    executions,
    grade,
    optimize,
    replay,
    summarize,
)
from .observation_review_study import paired
from .reading_pairs import MatchedExamples
from .registered import registered

DOMAINS = ("report", "distribution")
CALC = tool()
DIGIT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def functions(w, panel=None):
    return [CALC] if w["family"] in DOMAINS else catalog(w, panel)


def operand_check(expression, events):
    if not expression or not events or "value" not in events[-1]["observation"]:
        return False
    steps = events[-1]["call"]["arguments"]["steps"]
    # Change each distinct published operand in both programs and re-execute them.
    reference = {Fraction(m[0]) for m in DIGIT.finditer(expression)}
    if not reference:
        return False
    for value in reference:

        def perturb(m, chosen=value):
            return "(" + m[0] + "+11/7)" if Fraction(m[0]) == chosen else m[0]

        try:
            wanted = CALC.callback([DIGIT.sub(perturb, expression)])["value"]
            actual = CALC.callback([DIGIT.sub(perturb, s) for s in steps])["value"]
        except (ValueError, KeyError, ZeroDivisionError, SyntaxError):
            return False
        if Fraction(wanted) != Fraction(actual):
            return False
    return True


def score(w, i, tr):
    family = w["family"]
    events = executions(tr)
    if family in ("packages", "files"):
        action = w["calls"][i]
        gold = w["answers"][i]
        resource = next(iter(action["arguments"].values()))
        meaning = visible_reference(
            tr["answer"], resource, gold, file=family == "files"
        )
        valid = [
            e["call"] == action and str(next(iter(e["observation"].values()))) == gold
            for e in events
        ]
        done = tr["status"] == "answered"
        return {
            "success": done and meaning and valid == [True],
            "correct_answer": meaning,
            "completed": done,
            "path": valid == [True],
        }
    if family == "public":
        return grade(w, i, tr)
    clean = tr["answer"] or ""
    if family == "distribution":
        for line in w["questions"][0].split("\n\n", 1)[0].splitlines()[2:]:
            clean = clean.replace(line.split(" | ", 1)[0], "")
    grade_numeric = program_questions.score(w, dict(tr, answer=clean, turns=events))
    strict = (
        grade_numeric["success"]
        if w["expression"]
        else grade_numeric["correct_answer"] and grade_numeric["completed"]
    )
    return {
        **grade_numeric,
        "success": strict,
        "calculator_executed": bool(events),
        "bound_operands": grade_numeric["success"]
        and operand_check(w["expression"], events),
    }


def outcome(panel, runs):
    groups = {}
    reports = {}
    details = {}
    costs = {}
    for side, arm in runs.items():
        groups[side] = {
            k: summarize([r for r in arm if k == "all" or r["family"] == k])
            for k in ("all", *DOMAINS, "files", "packages", "public")
        }
        clusters = defaultdict(list)
        acc = defaultdict(lambda: defaultdict(int))
        for w, r in zip(panel["rows"], arm, strict=True):
            if w["family"] not in DOMAINS:
                continue
            s = r["requests"][0]["score"]
            fam = w["family"]
            clusters[(fam, w["cluster"])].append(s["success"])
            c = acc[fam + (" arithmetic" if w["expression"] else " lookup")]
            for k in (
                "correct_answer",
                "success",
                "calculator_executed",
                "bound_operands",
                "completed",
            ):
                c[k] += s[k]
        reports[side] = {
            f: {
                cluster
                for (fam, cluster), vs in clusters.items()
                if f == fam and len(vs) == 2 and all(vs)
            }
            for f in DOMAINS
        }
        details[side] = {k: dict(v) for k, v in acc.items()}
        allq = [q for r in arm for q in r["requests"]]
        ts = [t for q in allq for t in q["trajectory"]["turns"]]
        costs[side] = {
            "opening_tps": sum(
                len(q["trajectory"]["turns"][0]["prompt_ids"]) for q in allq
            )
            / sum(q["first_s"] for q in allq),
            "decode_tps": sum(len(t["token_ids"]) - 1 for t in ts)
            / sum(t["seconds"] - t["ttft_s"] for t in ts),
        }
    a, b = groups["candidate"], groups["control"]
    comparisons = {}
    for fam in DOMAINS:
        x, y = reports["candidate"][fam], reports["control"][fam]
        comparisons[fam] = {"loop": len(x), "base": len(y), **paired(x, y)}
    m = comparisons["report"]
    n = comparisons["distribution"]
    gates = {
        "reports": m["loop"] - m["base"] >= 4 and m["p"] <= 0.05,
        "operations": all(
            details["candidate"][k]["correct_answer"]
            >= details["control"][k]["correct_answer"]
            for k in details["candidate"]
        ),
        "field": n["loop"] > n["base"],
        "retention": all(
            a[k]["strict"] >= b[k]["strict"] for k in ("public", "files", "packages")
        ),
        "wall": a["all"]["wall_s"] < b["all"]["wall_s"],
        "first": a["all"]["first_s"] < b["all"]["first_s"],
        "peak": a["all"]["peak"] < b["all"]["peak"],
        "opening_rate": costs["candidate"]["opening_tps"]
        > costs["control"]["opening_tps"],
        "decode_rate": costs["candidate"]["decode_tps"]
        > costs["control"]["decode_tps"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": groups,
        "pairs": comparisons,
        "operations": details,
        "rates": costs,
    }


def verify(root, d):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    train, panel = (read(spec[k]) for k in ("training", "material"))
    fit = read(root / "fit.json")
    tok = load_tokenizer(local_snapshot())
    assert (fit["updates"], fit["passes"], fit["trainable"], fit["turns"]) == (
        spec["updates"],
        2,
        802816,
        len(train["examples"]),
    )
    assert fit["training"] == digest(Path(spec["training"])) and fit[
        "weights"
    ] == digest(root / "tool-adapters.safetensors")
    for pair in train["pairs"]:
        direct, calc = pair["questions"]
        assert direct["expression"] is None and Fraction(
            CALC.callback([calc["expression"]])["value"]
        ) == Fraction(calc["gold"])
    bank = [(None, 0, token_pair(tok, r)[1]) for r in train["examples"]]
    pairs = MatchedExamples(
        train["examples"], bank, tok, spec["pairs_per_batch"], spec["batch"]
    )
    rng = random.Random(spec["fit_seed"])
    assert fit["targets"] == sum(
        sum(len(s[2]) for s in pairs.sample(rng)) for _ in range(spec["updates"])
    )
    runs = {}
    for side in ("candidate", "control"):
        arm = runs[side] = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel["rows"], arm, strict=True):
            assert (w["id"], w["family"]) == (r["id"], r["family"])
            history = []
            for i, (question, q) in enumerate(
                zip(w["questions"], r["requests"], strict=True)
            ):
                if i == 0:
                    assert q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                history.append({"role": "user", "content": question})
                replay(tok, w, functions(w, panel), history, q, spec, side)
                assert score(w, i, q["trajectory"]) == q["score"]
    assert outcome(panel, runs) == d


def run(name):
    spec = registered(name)
    release = Path(spec["requires"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    assert all(
        digest(Path(spec["checkpoint"]) / p) == v
        for p, v in read(release / "export-files.json").items()
    )
    panel = read(spec["material"])
    assert package_tools.snapshot() == panel["environment"] and all(
        digest(Path(k)) == v["sha256"] for k, v in panel["files"].items()
    )
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            optimize(spec, root, budget)
        runs = {}
        for side in ("candidate", "control"):
            gc.collect()
            mx.clear_cache()
            dest = root / (side + ".json.gz")
            if not dest.exists():
                factory = (
                    (
                        lambda: Assistant(
                            spec["checkpoint"], tool_adapter=root, tool_format="qwen"
                        )
                    )
                    if side == "candidate"
                    else (lambda: DenseAssistant(True, "qwen"))
                )
                packed(
                    dest,
                    evaluate(
                        spec, panel, side, budget, factory, functions, scorer=score
                    ),
                )
            runs[side] = unpack(dest)
        save(root / "decision.json", outcome(panel, runs))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
