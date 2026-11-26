"""Execute a frozen two-model tool comparison under a native research lease."""

import argparse
import gc
import gzip
import json
from pathlib import Path

import mlx.core as mx

from loopqwen.deployment import load_trained
from loopqwen.evaluation.language7 import base
from loopqwen.evaluation.statistics import exact_mcnemar
from loopqwen.io import read, save
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.research.tool_worlds import score, tools_for
from loopqwen.tool_session import run


def summarize(records, spec):
    metrics = {}
    for arm, rows in records.items():
        metrics[arm] = {
            key: sum(r["score"][key] for r in rows) for key in rows[0]["score"]
        }
        metrics[arm].update(
            seconds=sum(r["trajectory"]["seconds"] for r in rows),
            tokens=sum(
                len(t["token_ids"]) for r in rows for t in r["trajectory"]["turns"]
            ),
            peak_bytes=max(
                t["peak_bytes"] for r in rows for t in r["trajectory"]["turns"]
            ),
        )
    pairs = list(zip(records["candidate"], records["control"], strict=True))
    gains = sum(a["score"]["success"] and not b["score"]["success"] for a, b in pairs)
    losses = sum(b["score"]["success"] and not a["score"]["success"] for a, b in pairs)
    p = exact_mcnemar(gains, losses)
    passed = (
        gains - losses >= spec["net_floor"]
        and p <= spec["p_max"]
        and metrics["candidate"]["seconds"]
        <= metrics["control"]["seconds"] * spec["time_ratio"]
    )
    for kind, minimum in spec.get("family_minimum", {}).items():
        items = read(spec["material"])
        successes = sum(
            row["score"]["success"]
            for row, item in zip(records["candidate"], items, strict=True)
            if item["kind"] == kind
        )
        passed = passed and successes >= minimum
    if spec.get("retention_floors"):
        items = read(spec["material"])
        for panel, criteria in spec["retention_floors"].items():
            paired = [
                (r, i)
                for r, i in zip(records["candidate"], items, strict=True)
                if i["panel"] == panel
            ]
            for family, floor in criteria.items():
                count = sum(
                    r["score"]["success"]
                    for r, i in paired
                    if family == "all" or i["kind"] == family
                )
                passed = passed and count >= floor
        passed = (
            passed
            and metrics["candidate"]["peak_bytes"] <= metrics["control"]["peak_bytes"]
        )
    details = {}
    if spec.get("answer_floor") is not None:
        rows = read(spec["material"])["rows"]
        changes = [
            int(a["score"]["correct_answer"]) - int(b["score"]["correct_answer"])
            for a, b in pairs
        ]
        by_source = {
            s: sum(x for x, r in zip(changes, rows, strict=True) if r["source"] == s)
            for s in {r["source"] for r in rows}
        }
        content_p = exact_mcnemar(changes.count(1), changes.count(-1))
        details["answer_comparison"] = dict(
            gains=changes.count(1),
            losses=changes.count(-1),
            p_exact=content_p,
            net_by_source=by_source,
        )
        passed = (
            passed
            and sum(changes) >= spec["answer_floor"]
            and content_p <= spec["p_max"]
            and all(x >= 0 for x in by_source.values())
            and metrics["candidate"]["peak_bytes"] <= metrics["control"]["peak_bytes"]
        )
    return dict(
        **details,
        passed=passed,
        metrics=metrics,
        gains=gains,
        losses=losses,
        p_exact=p,
        items=len(pairs),
    )


def workload(spec, *, replay=False):
    material = read(spec["material"])
    if spec.get("workload") == "arithmetic_programs":
        from loopqwen.evaluation import program_questions as arithmetic

        return material["rows"], arithmetic.tools_for, arithmetic.score
    if spec.get("workload") == "package_dependencies":
        from loopqwen.evaluation import package_questions

        return package_questions.workload(material, replay)
    if spec.get("workload") == "table_questions":
        from loopqwen.evaluation import table_questions

        items = [
            dict(r, user_request=r["question"] + table_questions.INSTRUCTION)
            for r in material["rows"]
        ]
        return items, table_questions.tools_for, table_questions.score
    if spec.get("workload") == "bfcl_exec":
        from loopqwen.evaluation import executable_tools as external

        items = [
            dict(
                row,
                user_request=row["question"][0][0]["content"] + external.INSTRUCTION,
            )
            for row in material["rows"]
        ]
        return items, external.tools_for, external.score
    return (
        [dict(row, user_request=row["question"]) for row in material],
        tools_for,
        score,
    )


def load_arm(arm, spec, folder, budget):
    if arm == "base":
        return base()[:2]
    if arm == "instruct":
        from mlx_lm.utils import load_tokenizer
        from loopqwen.model_loader import local_snapshot
        from loopqwen.research.projected_distillation import tutor

        model, _ = tutor({"teacher": spec["donor"]})
        return model, load_tokenizer(local_snapshot())
    if arm == "release02":
        return load_trained("research/results/native-language-release-02/model")[:2]
    from loopqwen.research.tool_fit import fitted

    return fitted(arm, spec, folder, budget)


def main(name):
    spec = registered(name)
    if "requires" in spec:
        from loopqwen.io import digest

        prerequisite = Path(spec["requires"])
        if (
            digest(prerequisite / "checksums.json") != spec["prerequisite_sha256"]
            or not read(prerequisite / "decision.json")["passed"]
        ):
            raise ValueError("Tool study prerequisite failed")
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise ValueError("Completed run exists")
    out.mkdir(parents=True, exist_ok=True)
    items, catalog, judge = workload(spec)
    budget = Budget(name, spec["limits"])
    records = {}
    try:
        for side in ("candidate", "control"):
            model, tok = load_arm(spec[side], spec, out, budget)
            mx.reset_peak_memory()
            arm = []
            for row in items:
                budget.check()
                traj = run(
                    model,
                    tok,
                    row["user_request"],
                    catalog(row),
                    max_turns=spec["max_turns"],
                    stop_at_turn_end=spec.get("stop_at_turn_end", False),
                    max_tokens=row.get("generation_cap", spec["max_tokens"]),
                    progress=lambda _: budget.check(),
                )
                arm.append(dict(id=row["id"], trajectory=traj, score=judge(row, traj)))
                print(
                    side,
                    row["id"],
                    traj["status"],
                    traj["answer"],
                    arm[-1]["score"]["success"],
                    flush=True,
                )
            (out / (side + ".json.gz")).write_bytes(
                gzip.compress(json.dumps(arm).encode(), mtime=0)
            )
            records[side] = arm
            del model, tok
            gc.collect()
            mx.clear_cache()
        assert all(
            a["trajectory"]["turns"][0]["prompt_ids"]
            == b["trajectory"]["turns"][0]["prompt_ids"]
            for a, b in zip(records["candidate"], records["control"], strict=True)
        )
        decision = summarize(records, spec)
        save(out / "decision.json", decision)
        print(decision, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    main(parser.parse_args().name)
