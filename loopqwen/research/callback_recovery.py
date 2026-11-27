"""Fit only on authentic failed calls, then run open conversations on real field requests."""

import argparse
import gc
import json
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..calculator_tool import calculate, tool
from ..callback_recovery import RecoveryAssistant
from ..evaluation.mechanism import packed, unpack
from ..evaluation.program_questions import score as arithmetic_score
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_session import compact, initial_messages
from ..tool_session import run as conversation
from .core import Budget
from .folded_system import judge, old_scores, tools_for
from .mixed_field import replay_exchange
from .registered import registered
from .tool_fit import features, fit, restore


def curriculum(spec, inputs, attempts):
    assert len(inputs) == len(attempts) == spec["collection_size"]
    lessons = []
    for w, r in zip(inputs, attempts, strict=True):
        assert w["id"] == r["id"]
        t = r["trajectory"]["turns"][0]
        if "error" not in t.get("observation", {}):
            continue
        msgs = initial_messages(w["user"], [tool()])
        msgs.extend(
            [
                {"role": "assistant", "content": t["visible"]},
                {"role": "tool", "content": compact(t["observation"])},
            ]
        )
        target = json.loads(w["target"])
        result = calculate(**target["arguments"])
        assert target["name"] == "calculate" and Fraction(result["value"]) == Fraction(
            w["gold"]
        )
        lessons.append({"messages": list(msgs), "visible": w["target"], "thinking": ""})
        msgs.extend(
            [
                {"role": "assistant", "content": w["target"]},
                {"role": "tool", "content": compact(result)},
            ]
        )
        lessons.append({"messages": msgs, "visible": w["gold"], "thinking": ""})
    return {"examples": lessons}


def collect(spec, root, net, tok, inputs, budget):
    path = root / "opening-attempts.json.gz"
    rows = unpack(path) if path.exists() else []
    assert [w["id"] for w in inputs[: len(rows)]] == [r["id"] for r in rows]
    for w in inputs[len(rows) :]:
        tr = conversation(
            net,
            tok,
            w["user"],
            [tool()],
            max_turns=1,
            max_tokens=spec["max_tokens"],
            progress=lambda _: budget.check(),
        )
        rows.append({"id": w["id"], "trajectory": tr})
        if len(rows) % 8 == 0 or len(rows) == len(inputs):
            packed(path, rows)
            print("opening attempts", len(rows), flush=True)
    return rows


def failed_collection(lessons, spec):
    n = len(lessons["examples"]) // 2
    return {
        "passed": False,
        "status": "insufficient_real_callback_failures",
        "collected": spec["collection_size"],
        "eligible": n,
    }


def fit_recovery(spec, root, budget):
    parent = Path(spec["public"])
    assert (
        digest(parent / "checksums.json") == spec["public_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    net, tok = restore(parent)
    inputs = read(spec["training"])["rows"]
    attempts = collect(spec, root, net, tok, inputs, budget)
    lessons = curriculum(spec, inputs, attempts)
    if len(lessons["examples"]) < 2 * spec["minimum_failures"]:
        return failed_collection(lessons, spec)
    if not (root / "fit.json").exists():
        mx.random.seed(spec["fit_seed"])
        fit(
            net,
            tok,
            spec,
            root,
            budget,
            prepare=lambda m, t, _rows, b: features(m, t, lessons, b),
        )
    return None


def grading(w, tr, tok):
    score = judge(w, tr, tok)
    if w["cohort"] != "retention":
        score["success"] = arithmetic_score(w, tr)["success"]
    return score


def evaluate(spec, root, panel, field, side, budget):
    actor = None
    arm = []
    gc.collect()
    mx.clear_cache()
    for w in panel:
        catalog = tools_for(w, field)
        mx.synchronize()
        mx.reset_peak_memory()
        at = perf_counter()
        if actor is None:
            actor = (
                RecoveryAssistant(spec["checkpoint"], root)
                if side == "candidate"
                else DenseAssistant(True)
            )
        first = []
        expected_tokens = len(w["initial_ids"])

        def progress(n, expected=expected_tokens, stamps=first, start=at):
            if not stamps and n == expected:
                stamps.append(perf_counter() - start)
            budget.check()

        tr = actor.respond(
            w.get("user_request", w["question"]),
            catalog,
            max_tokens=spec["max_tokens"],
            max_turns=spec["max_turns"],
            progress=progress,
            reuse_prefix=True,
            record_kv_history=True,
        )
        mx.synchronize()
        wall = perf_counter() - at
        assert len(first) == 1 and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
        arm.append(
            {
                "id": w["id"],
                "trajectory": tr,
                "score": grading(w, tr, actor.tokenizer),
                "wall_s": wall,
                "first_s": first[0],
                "peak": mx.get_peak_memory(),
            }
        )
        if len(arm) % 12 == 0:
            packed(root / (side + ".json.gz"), arm)
            print(side, len(arm), flush=True)
    return arm


def totals(rows):
    ts = [t for r in rows for t in r["trajectory"]["turns"]]
    return {
        **{
            k: sum(bool(r["score"][k]) for r in rows)
            for k in ("grounded", "answer", "completed", "success")
        },
        "seconds": sum(r["wall_s"] for r in rows),
        "first": sum(r["first_s"] for r in rows),
        "peak": max(r["peak"] for r in rows),
        "tokens": sum(len(t["token_ids"]) for t in ts),
        "actual_errors": sum("error" in t.get("observation", {}) for t in ts),
        "used_recovery": sum(
            any(t.get("recovery_active", False) for t in r["trajectory"]["turns"])
            for r in rows
        ),
        "successful_after_error": sum(
            bool(r["score"]["grounded"])
            and any(
                "error" in t.get("observation", {}) for t in r["trajectory"]["turns"]
            )
            for r in rows
        ),
    }


def differences(a, b, key):
    ds = [
        int(x["score"][key]) - int(y["score"][key]) for x, y in zip(a, b, strict=True)
    ]
    return {
        "gains": ds.count(1),
        "losses": ds.count(-1),
        "p": exact_mcnemar(ds.count(1), ds.count(-1)),
    }


def same_turn(a, b):
    return all(
        a.get(k) == b.get(k)
        for k in ("prompt_ids", "token_ids", "visible", "call", "observation")
    )


def outcome(spec, panel, data, eligible):
    prior = old_scores(spec)
    reference = {}
    for key in ("public", "field"):
        for r in unpack(Path(spec[key]) / "candidate.json.gz"):
            reference[r["id"]] = r["trajectory"]
    earlier = [
        w["id"]
        for w, r in zip(panel, data["candidate"], strict=True)
        if prior.get(w["id"]) and not r["score"]["success"]
    ]
    stable = []
    for w, r in zip(panel, data["candidate"], strict=True):
        old = reference.get(w["id"])
        if old is None:
            continue
        turns = old["turns"]
        for i, t in enumerate(turns):
            if i >= len(r["trajectory"]["turns"]) or not same_turn(
                r["trajectory"]["turns"][i], t
            ):
                stable.append(w["id"])
                break
            if w["family"] == "arithmetic" and "error" in t.get("observation", {}):
                break
        else:
            if len(r["trajectory"]["turns"]) != len(turns):
                stable.append(w["id"])
    groups = {}
    for name in (
        "all",
        "fresh",
        "ASDiv",
        "SVAMP",
        "retention",
        *sorted({w["family"] for w in panel if w["cohort"] == "retention"}),
    ):
        ix = [
            i
            for i, w in enumerate(panel)
            if name in ("all", w["cohort"])
            or name == "fresh"
            and w["cohort"] != "retention"
            or w["cohort"] == "retention"
            and name == w["family"]
        ]
        a, b = [[data[side][i] for i in ix] for side in ("candidate", "control")]
        groups[name] = {
            "n": len(ix),
            "candidate": totals(a),
            "control": totals(b),
            "strict_pairs": differences(a, b, "success"),
            "answer_pairs": differences(a, b, "answer"),
        }
    fresh = groups["fresh"]
    all_ = groups["all"]
    a, b = all_["candidate"], all_["control"]
    fa, fb = fresh["candidate"], fresh["control"]
    passed = (
        not earlier
        and not stable
        and fa["grounded"] >= fb["grounded"] + spec["grounded_gain"]
        and fresh["strict_pairs"]["p"] <= 0.05
        and fa["answer"] >= fb["answer"] + spec["answer_gain"]
        and fresh["answer_pairs"]["p"] <= 0.05
        and fa["successful_after_error"] >= spec["recovery_floor"]
        and all(
            groups[k]["candidate"]["answer"] >= groups[k]["control"]["answer"]
            for k in ("ASDiv", "SVAMP")
        )
        and a["success"] > b["success"]
        and all_["strict_pairs"]["p"] <= 0.05
        and all(a[k] <= b[k] for k in ("seconds", "first", "peak"))
    )
    return {
        "passed": bool(passed),
        "eligible_training_errors": eligible,
        "groups": groups,
        "historical_losses": earlier,
        "changed_before_error_or_on_unaffected_requests": stable,
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    tok = load_tokenizer(local_snapshot())
    training = read(spec["training"])
    assert digest(Path(training["source"])) == training["source_sha256"]
    targets = {w["id"]: w for w in read(training["source"])["examples"]}
    inputs = training["rows"]
    attempts = unpack(root / "opening-attempts.json.gz")
    for w, r in zip(inputs, attempts, strict=True):
        teacher = targets[w["id"]]
        assert (
            w["target"] == teacher["visible"]
            and w["user"] == teacher["messages"][-1]["content"]
        )
        assert (
            w["gold"] == targets[w["id"].removesuffix("-call") + "-result"]["visible"]
        )
        replay_exchange(
            tok,
            [tool()],
            initial_messages(w["user"], [tool()]),
            r["trajectory"],
            dict(spec, max_turns=1),
        )
    lessons = curriculum(spec, inputs, attempts)
    eligible = len(lessons["examples"]) // 2
    if eligible < spec["minimum_failures"]:
        assert (
            decision == failed_collection(lessons, spec)
            and not (root / "fit.json").exists()
        )
        return
    meta = read(root / "fit.json")
    assert (
        meta["weights"] == digest(root / "tool-adapters.safetensors")
        and meta["weights"] != read(Path(spec["public"]) / "fit.json")["weights"]
    )
    assert (
        meta["passes"] == 2
        and meta["turns"] == 2 * eligible
        and meta["updates"] == spec["updates"]
        and meta["trainable"] == 802816
        and meta["training"] == digest(Path(spec["training"]))
    )
    panel = read(spec["material"])["rows"]
    field = read(spec["field_data"])
    data = {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel, data[side], strict=True):
            catalog = tools_for(w, field, True)
            user = w.get("user_request", w["question"])
            messages = (
                initial_messages(user, catalog)
                if catalog
                else [{"role": "user", "content": user}]
            )
            tr = r["trajectory"]
            assert (
                r["id"] == w["id"]
                and w["initial_ids"] == tr["turns"][0]["prompt_ids"]
                and 0 < r["first_s"] <= r["wall_s"]
                and r["peak"] > 0
            )
            replay_exchange(tok, catalog, messages, tr, spec)
            assert r["score"] == grading(w, tr, tok)
            previous = False
            prior_used = False
            for t in tr["turns"]:
                if side == "candidate" and catalog:
                    assert t["recovery_active"] == previous
                    kv = t["kv_history"]
                    assert {(v["layer"], v["turn"]) for v in kv} == {
                        (i, 0) for i in (3, 7, 11, 15, 19, 23)
                    } | {(23, 1)}
                    assert all(
                        v["offset"]
                        == v["retained"]
                        == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                        for v in kv
                    )
                    if previous and not prior_used:
                        assert t["reused_prompt_tokens"] == 0
                    prior_used = previous
                    previous |= w["family"] == "arithmetic" and "error" in t.get(
                        "observation", {}
                    )
    assert outcome(spec, panel, data, eligible) == decision


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    assert not (root / "decision.json").exists()
    release = Path(spec["release"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["release_sha256"]
    )
    checkpoint = Path(spec["checkpoint"])
    assert (
        digest(checkpoint / "tool-export-files.json")
        == digest(release / "export-files.json")
        == spec["bundle_inventory"]
    )
    assert all(
        digest(checkpoint / k) == v
        for k, v in read(release / "export-files.json").items()
    )
    panel = read(spec["material"])["rows"]
    field = read(spec["field_data"])
    from ..package_tools import snapshot

    assert snapshot() == field["environment"] and all(
        digest(Path(k)) == v["sha256"] for k, v in field["files"].items()
    )
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        stopped = fit_recovery(spec, root, budget)
        if stopped is not None:
            save(root / "decision.json", stopped)
            return
        data = {}
        for side in ("candidate", "control"):
            dest = root / (side + ".json.gz")
            arm = unpack(dest) if dest.exists() else []
            if len(arm) != len(panel):
                arm = evaluate(spec, root, panel, field, side, budget)
                packed(dest, arm)
            data[side] = arm
        d = outcome(spec, panel, data, read(root / "fit.json")["turns"] // 2)
        save(root / "decision.json", d)
        print("qualified", d["passed"], flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
