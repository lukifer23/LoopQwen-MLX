"""Execute a frozen two-model tool comparison under a native research lease."""

import argparse
import sys
import gc
import gzip
import json
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        if isinstance(items, dict):
            items = items["rows"]
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
    if spec.get("observation_adapter"):
        from loopqwen.evaluation.mechanism import unpack

        prior = unpack(Path(spec["observation_adapter"]) / "candidate.json.gz")
        first_equal = True
        joint_losses = 0
        for (a, b), ref in zip(pairs, prior, strict=True):
            if a["id"] != ref["id"]:
                raise ValueError("Different joint reference request")
            first_equal &= (
                a["trajectory"]["turns"][0]["token_ids"]
                == b["trajectory"]["turns"][0]["token_ids"]
            )
            joint_losses += int(ref["score"]["success"] and not a["score"]["success"])
        passed = (
            passed
            and first_equal
            and losses == 0
            and joint_losses == 0
            and metrics["candidate"]["peak_bytes"]
            <= metrics["control"]["peak_bytes"] + spec["extra_bytes"]
        )
        details["observation_policy"] = dict(
            first_turns_equal=first_equal,
            lost_from_joint=joint_losses,
            lost_from_original=losses,
        )

    if spec.get("matched_retention"):
        from loopqwen.evaluation.mechanism import unpack
        from scripts.tool_release import tokens

        material = read(spec["material"])["rows"]
        groups = {}
        for panel in {r["panel"] for r in material}:
            ps = [
                p for p, r in zip(pairs, material, strict=True) if r["panel"] == panel
            ]
            groups[panel] = [
                sum(int(x[j]["score"]["success"]) for x in ps) for j in (0, 1)
            ]
        original = unpack(Path(spec["requires"]) / "candidate.json.gz")
        exact = all(
            r["id"] == ref["id"]
            and tokens(r["trajectory"]) == tokens(ref["trajectory"])
            for r, ref in zip(records["candidate"], original, strict=True)
        )
        passed = (
            exact
            and gains >= losses
            and all(a >= b - 1 for a, b in groups.values())
            and metrics["candidate"]["seconds"]
            <= metrics["control"]["seconds"] * spec["time_ratio"]
            and metrics["candidate"]["peak_bytes"]
            <= metrics["control"]["peak_bytes"] + spec["extra_bytes"]
        )
        details["matched"] = dict(
            panels=groups,
            exact=exact,
            causal_gain=passed
            and gains - losses >= spec["net_floor"]
            and p <= spec["p_max"],
        )

    if spec.get("workload") == "sql_transfer_retention":
        groups = read(spec["material"])["rows"]
        metrics_by_group = {}
        for side, rs in records.items():
            grouped = {
                key: [0, 0]
                for key in (
                    "public",
                    "value",
                    "maximum",
                    "minimum",
                    "count",
                    "sum",
                    "average",
                )
            }
            for r, w in zip(rs, groups, strict=True):
                group = (
                    ("value", "maximum", "minimum", "count", "sum", "average")[
                        w["sql"]["agg"]
                    ]
                    if w["provider"] == "wikisql"
                    else "public"
                )
                grouped[group][0] += r["score"]["success"]
                grouped[group][1] += 1
            metrics_by_group[side] = grouped
        diff = [
            int(a["score"]["success"]) - int(b["score"]["success"])
            for (a, b), w in zip(pairs, groups, strict=True)
            if w["provider"] == "wikisql"
        ]
        sql_p = exact_mcnemar(diff.count(1), diff.count(-1))
        details["table_transfer"] = dict(
            by_group=metrics_by_group, net=sum(diff), p_exact=sql_p
        )
        passed = (
            passed
            and sum(diff) >= spec["sql_net_floor"]
            and sql_p <= spec["p_max"]
            and all(
                v[0] >= spec["public_floor" if k == "public" else "operation_floor"]
                for k, v in metrics_by_group["candidate"].items()
            )
            and metrics["candidate"]["peak_bytes"] <= metrics["control"]["peak_bytes"]
        )
    if spec.get("workload") == "schema_actions":
        from loopqwen.evaluation.mechanism import unpack
        from scripts.tool_release import tokens

        items = read(spec["material"])["rows"]
        baseline = {
            r["id"]: r["trajectory"]
            for r in unpack(Path(spec["requires"]) / "candidate.json.gz")
        }
        groups = {}
        for provider in ("bfcl_exec", "wikisql"):
            sub = [
                pair
                for pair, item in zip(pairs, items, strict=True)
                if item["provider"] == provider
            ]
            groups[provider] = {
                s: sum(pair[j]["score"]["success"] for pair in sub)
                for j, s in enumerate(records)
            }
        stock_parity = all(
            tokens(b["trajectory"]) == tokens(baseline[b["id"]])
            for b in records["control"]
            if b["id"] in baseline
        )
        invalid = {
            s: sum(r["trajectory"]["status"] == "invalid_call" for r in rs)
            for s, rs in records.items()
        }
        corrections = sum(
            t["replaced_by_schema"]
            for r in records["candidate"]
            for t in r["trajectory"]["turns"]
        )
        details["schema"] = dict(
            by_provider=groups,
            invalid=invalid,
            replaced=corrections,
            stock_parity=stock_parity,
        )
        passed = (
            passed
            and stock_parity
            and corrections > 0
            and invalid["candidate"] < invalid["control"]
            and all(g["candidate"] > g["control"] for g in groups.values())
            and metrics["candidate"]["peak_bytes"]
            <= metrics["control"]["peak_bytes"] + spec["extra_bytes"]
        )
    if spec.get("workload") == "venue_dialogues":
        from loopqwen.evaluation.venue_dialogues import paired_result

        pairs_score = paired_result(read(spec["material"]), records)
        details["venue_dialogues"] = pairs_score
        passed = (
            passed
            and pairs_score["wins"] - pairs_score["losses"] >= spec["dialogue_floor"]
            and pairs_score["p"] <= spec["p_max"]
            and all(
                value >= spec["service_floors"][service]
                for service, value in pairs_score["both_by_service"][
                    "candidate"
                ].items()
            )
            and pairs_score["public_tools"]["candidate"] >= spec["public_floor"]
            and metrics["candidate"]["peak_bytes"] <= metrics["control"]["peak_bytes"]
        )
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
    if spec.get("workload") == "mixed_tool_retention":
        from loopqwen.evaluation import executable_tools as external

        items = [
            dict(
                r,
                user_request=r["question"][0][0]["content"] + external.INSTRUCTION
                if r["provider"] == "bfcl_exec"
                else r["question"],
                turn_cap=3 if r["provider"] == "bfcl_exec" else spec["max_turns"],
            )
            for r in material["rows"]
        ]

        def catalog(row):
            return (
                external.tools_for(row)
                if row["provider"] == "bfcl_exec"
                else tools_for(row)
            )

        def judge(row, trajectory):
            return (
                external.score(row, trajectory)
                if row["provider"] == "bfcl_exec"
                else score(row, trajectory)
            )

        return items, catalog, judge
    if spec.get("workload") in ("sql_transfer_retention", "schema_actions"):
        from loopqwen.evaluation import executable_tools, table_questions

        items = []
        for r in material["rows"]:
            public = r["provider"] == "bfcl_exec"
            question = (
                r["question"][0][0]["content"] + executable_tools.INSTRUCTION
                if public
                else r["question"] + table_questions.INSTRUCTION
            )
            items.append(dict(r, user_request=question))

        def provider(r):
            return executable_tools if r["provider"] == "bfcl_exec" else table_questions

        return (
            items,
            lambda r: provider(r).tools_for(r),
            lambda r, t: provider(r).score(r, t),
        )
    if spec.get("workload") == "venue_dialogues":
        from loopqwen.evaluation import venue_dialogues

        return venue_dialogues.workload(material)
    if spec.get("workload") == "arithmetic_programs":
        from loopqwen.evaluation import program_questions as arithmetic

        items = [
            dict(r, user_request=r["question"] + arithmetic.INSTRUCTION)
            for r in material["rows"]
        ]
        if any(
            "user_request" in r and r["user_request"] != i["user_request"]
            for i, r in zip(items, material["rows"], strict=True)
        ):
            raise ValueError("Arithmetic prompt differs from its frozen rendering")
        return items, arithmetic.tools_for, arithmetic.score
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
            observation_model = None
            if side == "candidate" and spec.get("observation_adapter"):
                from loopqwen.tool_model import with_adapter

                new = Path(spec["observation_adapter"])
                old = Path(spec["candidate"].removeprefix("restore:"))
                reference = read(new / "registration.json")
                if (
                    digest(new / "checksums.json") != spec["observation_archive"]
                    or not read(new / "decision.json")["passed"]
                    or reference["initial_sha256"] != digest(old / "checksums.json")
                    or any(
                        reference[k] != read(old / "registration.json")[k]
                        for k in ("prefix_bits", "rank", "start_layer", "donor")
                    )
                ):
                    raise ValueError("Different tool view lineage")
                observation_model = with_adapter(
                    model,
                    new / "tool-adapters.safetensors",
                    read(new / "fit.json")["weights"],
                )
            mx.reset_peak_memory()
            arm = []
            for row in items:
                budget.check()
                traj = run(
                    model,
                    tok,
                    row["user_request"],
                    catalog(row),
                    max_turns=row.get("turn_cap", spec["max_turns"]),
                    stop_at_turn_end=spec.get("stop_at_turn_end", False),
                    max_tokens=row.get("generation_cap", spec["max_tokens"]),
                    progress=lambda _: budget.check(),
                    observation_model=observation_model,
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
            del model, tok, observation_model
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
