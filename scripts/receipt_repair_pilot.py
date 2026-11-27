"""Test free delivery from correct and misleading real calculations on reserved development reports."""

import argparse
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import re
from time import monotonic

import mlx.core as mx

from loopqwen.calculator_tool import calculate, tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.research.native_function_study import executions
from loopqwen.research.source_copy_transfer import displayed, SUFFIXES
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.tool_session import run as dialogue
from scripts.crossed_numeric_sources import DIR
from scripts.receipt_repair_fit import OUT as FIT, PLAN as TRAIN, PARENT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/receipt-repair-pilot.json")
OUT = Path("docs/measurements/receipt-repair-pilot")


def reference(r):
    m = re.fullmatch(
        r"(-?[\d,]+(?:\.\d+)?)\s*(percent|million|billion|thousand)?", r["reference"]
    )
    if not m:
        raise ValueError("Unrecognized human target")
    text = m[1].replace(",", "")
    unit = m[2] or ""
    decimals = len(text.partition(".")[2])
    tolerance = Fraction(1, 2 * 10**decimals)
    value = Fraction(r["true"]["value"])
    assert abs(Fraction(text) - value) <= tolerance
    return value, unit, tolerance


def evaluate(row, tr):
    ref, unit, tol = reference(row)
    got = displayed(tr["answer"] or "")
    delivered = False
    if got:
        value, suffix = got
        is_percent = suffix == "%" or suffix.startswith("percent")
        if unit == "percent":
            delivered = is_percent and abs(value - ref) <= tol
        elif not is_percent:
            delivered = (
                abs(
                    value
                    * Fraction(10) ** (SUFFIXES.get(suffix, 0) - SUFFIXES.get(unit, 0))
                    - ref
                )
                <= tol
            )
    events = executions(tr)
    for event in events:
        assert event["observation"] == calculate(event["call"]["arguments"]["steps"])
    valid_final = (
        bool(events)
        and "value" in events[-1]["observation"]
        and Fraction(events[-1]["observation"]["value"]) == ref
    )
    correct = tr["status"] == "answered" and delivered
    repaired = valid_final if row["condition"] == "crossed" else not events
    return dict(
        correct=bool(correct),
        appropriate=bool(repaired),
        success=bool(correct and repaired),
        executions=len(events),
    )


def prepare():
    rows = unpack(DIR / "training.json.gz")
    for r in rows:
        reference(r)
        assert calculate(r["correct_steps"]) == r["true"]
    pilots = set(read(DIR / "inventory.json")["pilot_reports"])
    held = [r for r in rows if r["report"] in pilots]
    assert len(held) == 64
    return held


def run(freeze):
    held = prepare()
    plan = dict(
        fit=digest(TRAIN),
        data=digest(DIR / "training.json.gz"),
        implementation=digest(Path(__file__)),
        prompt_hashes=[r["input_sha256"] for r in held],
        budget_s=600,
        peak_bytes=3 * 2**30,
        protocol="Training diagnostic on the 16 original pilot reports reserved from this intervention, two real questions each under both the correct and a crossed valid calculator result. Run only if registered fitting gates pass. Use the actual two-read fitted weights and no other model. Native greedy reasoning enabled, 256 tokens per turn, up to three turns, identical ordered questions and real callbacks. Require the correctly scaled and rounded delivered value. After a correct observation require direct delivery. After a misleading observation require an executed correction returning the exact human program result. At least eight questions must pass both histories, across at least six different reports, with at least 16 correct direct deliveries and at least 12 repairs. A failure closes held product evaluation for these weights.",
    )
    if freeze:
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    committed(FIT / "receipt.json")
    fit = read(FIT / "receipt.json")
    if not fit["passed"]:
        return
    out = []
    stop = None
    begin = monotonic()
    actor = SharedQueryAssistant(
        read(PARENT / "registration.json")["checkpoint"], FIT / "research-policy"
    )
    mx.reset_peak_memory()

    def guard(_=None):
        if (
            monotonic() - begin > plan["budget_s"]
            or mx.get_peak_memory() > plan["peak_bytes"]
        ):
            raise InterruptedError("native pilot limit")

    try:
        for r in held:
            guard()
            tic = monotonic()
            tr = dialogue(
                actor._tool_model,
                actor.tokenizer,
                None,
                [tool()],
                _messages=deepcopy(r["messages"]),
                max_tokens=256,
                max_turns=3,
                reuse_prefix=True,
                prefill_cache_state=True,
                tool_format="qwen",
                record_kv_history=True,
                progress=guard,
            )
            assert fingerprint(tr["turns"][0]["prompt_ids"]) == r["input_sha256"]
            out.append(
                dict(
                    id=r["question"],
                    report=r["report"],
                    condition=r["condition"],
                    trajectory=tr,
                    score=evaluate(r, tr),
                    seconds=monotonic() - tic,
                )
            )
            print("pilot histories", len(out), flush=True)
    except InterruptedError as e:
        stop = str(e)
    OUT.mkdir(parents=True, exist_ok=True)
    packed(OUT / "trajectories.json.gz", out)
    successes = {
        mode: {r["id"] for r in out if r["condition"] == mode and r["score"]["success"]}
        for mode in ("correct", "crossed")
    }
    pairs = successes["correct"] & successes["crossed"]
    reports = {r["report"] for r in out if r["id"] in pairs}
    gates = dict(
        complete=stop is None and len(out) == 64,
        direct=len(successes["correct"]) >= 16,
        repairs=len(successes["crossed"]) >= 12,
        pairs=len(pairs) >= 8,
        reports=len(reports) >= 6,
    )
    persist(
        OUT / "receipt.json",
        dict(
            plan=digest(PLAN),
            weights=fit["weights"],
            successes={k: sorted(v) for k, v in successes.items()},
            pairs=sorted(pairs),
            reports=sorted(reports),
            gates=gates,
            passed=all(gates.values()),
            stop=stop,
            seconds=monotonic() - begin,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
