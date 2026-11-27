"""Compare two free decoders on changed human questions, new arithmetic and real services."""

import argparse
import gc
import random
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import package_tools
from ..calculator_tool import tool
from ..evaluation import program_questions
from ..evaluation.mechanism import packed, unpack
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


def functions(w, panel=None):
    return [tool()] if w["family"] == "DROP" else catalog(w, panel)


def score(w, i, tr):
    if w["family"] != "DROP":
        return grade(w, i, tr)
    result = program_questions.score(w, dict(tr, turns=executions(tr)))
    return {
        **result,
        "grounded_arithmetic": result["success"],
        "success": result["completed"] and result["correct_answer"],
    }


def outcome(panel, data):
    groups, document, entire, numeric = {}, {}, {}, {}
    for side, arm in data.items():
        families = sorted({w["family"] for w in panel["rows"]})
        groups[side] = {
            k: summarize([r for r in arm if k == "all" or k == r["family"]])
            for k in ("all", *families)
        }
        by_cluster = defaultdict(list)
        doc = {}
        num = set()
        for w, r in zip(panel["rows"], arm, strict=True):
            ok = all(q["score"]["success"] for q in r["requests"])
            by_cluster[w["cluster"]].append(ok)
            if w["family"] == "DROP":
                doc[w["cluster"]] = True
            if w["family"] == "ASDiv" and r["requests"][0]["score"]["correct_answer"]:
                num.add(w["id"])
        entire[side] = {k for k, vs in by_cluster.items() if all(vs)}
        document[side] = entire[side] & doc.keys()
        numeric[side] = num
    comparisons = {
        k: paired(v["candidate"], v["control"])
        for k, v in [("documents", document), ("all", entire), ("arithmetic", numeric)]
    }
    a, b = groups["candidate"], groups["control"]
    gates = {
        "changed_questions": comparisons["documents"]["wins"]
        - comparisons["documents"]["losses"]
        >= 4
        and comparisons["documents"]["p"] <= 0.05,
        "arithmetic": comparisons["arithmetic"]["wins"]
        > comparisons["arithmetic"]["losses"]
        and a["ASDiv"]["grounded"] >= b["ASDiv"]["grounded"],
        "overall": comparisons["all"]["wins"] > comparisons["all"]["losses"]
        and comparisons["all"]["p"] <= 0.05,
        "functions": all(
            a[f]["strict"] >= b[f]["strict"] for f in ("public", "files", "packages")
        ),
        **{k: a["all"][k] <= b["all"][k] for k in ("wall_s", "first_s", "peak")},
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": groups,
        "paired": comparisons,
        "whole_passages": {k: len(v) for k, v in document.items()},
        "whole_clusters": {k: len(v) for k, v in entire.items()},
    }


def verify(root, d):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    train, panel = (read(spec[k]) for k in ("training", "material"))
    tok = load_tokenizer(local_snapshot())
    fit = read(root / "fit.json")
    assert fit["training"] == digest(Path(spec["training"])) and fit[
        "weights"
    ] == digest(root / "tool-adapters.safetensors")
    assert (fit["updates"], fit["trainable"], fit["passes"]) == (
        spec["updates"],
        802816,
        2,
    )
    bank = [(None, 0, token_pair(tok, r)[1]) for r in train["examples"]]
    matched = MatchedExamples(
        train["examples"], bank, tok, spec["pairs_per_batch"], spec["batch"]
    )
    assert len(matched.pairs) == len(train["pairs"])
    rng = random.Random(spec["fit_seed"])
    assert fit["targets"] == sum(
        sum(len(row[2]) for row in matched.sample(rng)) for _ in range(spec["updates"])
    )
    assert fit["turns"] == len(bank)
    for r in train["pairs"]:
        if r["kind"] == "SVAMP":
            assert (
                r["returns"] == [tool().callback([s])["value"] for s in r["programs"]]
                and len(set(r["returns"])) == 2
            )
    data = {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel["rows"], data[side], strict=True):
            assert (w["id"], w["family"]) == (r["id"], r["family"])
            assert (
                r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                == w["initial_ids"]
            )
            history = []
            for i, (question, q) in enumerate(
                zip(w["questions"], r["requests"], strict=True)
            ):
                history.append({"role": "user", "content": question})
                replay(tok, w, functions(w, panel), history, q, spec, side)
                assert score(w, i, q["trajectory"]) == q["score"]
    assert outcome(panel, data) == d


def run(name):
    spec = registered(name)
    release = Path(spec["requires"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    assert all(
        digest(Path(spec["checkpoint"]) / k) == v
        for k, v in read(release / "export-files.json").items()
    )
    train, panel = (read(spec[k]) for k in ("training", "material"))
    assert all(
        digest(Path(k)) == v
        for part in (train, panel)
        for k, v in part["sources"].items()
    )
    assert package_tools.snapshot() == panel["environment"] and all(
        digest(Path(k)) == v["sha256"] for k, v in panel["files"].items()
    )
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            optimize(spec, root, budget)
        data = {}
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
            data[side] = unpack(dest)
        save(root / "decision.json", outcome(panel, data))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
