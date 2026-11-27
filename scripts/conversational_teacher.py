"""Qualify numerical training traces from a larger, training-only native model."""

import argparse
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from time import monotonic

import mlx.core as mx
import psutil
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from loopqwen.calculator_tool import calculate, tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.looped_model import LoopedModel
from loopqwen.qwen_tools import render, declarations
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_study import executions
from loopqwen.research.report_calculation import operation_evidence
from loopqwen.tool_session import run as dialogue
from loopqwen.worker import acquire
from scripts.conversational_finance_sources import ROOT
from scripts.program_language_material import user, call, observed, answer
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/conversational-training-teacher.json")
OUT = Path("docs/measurements/conversational-training-teacher")
DATA = ROOT / "training.json.gz"
SYSTEM = "Use the calculator for numerical follow-ups. Think through which quantity the earlier turns establish, execute the calculation, and give the requested result in its source units."
PROVIDER = tool()


def select():
    rows = sorted(
        unpack(DATA), key=lambda w: fingerprint(["teacher-provenance-721", w["id"]])
    )
    selected = []
    seen = set()
    for w in rows:
        if w["issuer"] in seen:
            continue
        candidates = []
        for i, t in enumerate(w["turns"]):
            past = w["turns"][:i]
            value = Fraction(t["result"]["value"])
            if (
                "#0" in t["program"]
                and value != 0
                and all(Fraction(p["result"]["value"]) != value for p in past)
                and any(p["arithmetic"] and p["program"] in t["program"] for p in past)
            ):
                candidates.append(i)
        if candidates:
            i = min(candidates, key=lambda i: fingerprint([w["id"], i, "next"]))
            selected.append((w, i))
            seen.add(w["issuer"])
        if len(selected) == 24:
            break
    assert len(selected) == 24
    return selected


def inputs(w, i):
    history = [{"role": "system", "content": SYSTEM}]
    for j, t in enumerate(w["turns"][:i]):
        history.append(user((w["document"] + "\n\n" if j == 0 else "") + t["question"]))
        if t["arithmetic"]:
            r = calculate(t["steps"])
            assert r == t["result"]
            history.extend([call(PROVIDER, steps=t["steps"]), observed(r)])
        history.append(answer(t["release_value"]))
    return history, w["turns"][i]["question"]


def assessment(w, i, tr):
    current = w["turns"][i]
    expected = Fraction(current["result"]["value"])
    prior = {Fraction(t["result"]["value"]) for t in w["turns"][:i]}
    events = executions(tr)
    if len(events) != 1 or "value" not in events[0]["observation"]:
        return dict(accepted=False, calls=len(events))
    e = events[0]
    steps = e["call"]["arguments"]["steps"]
    result = calculate(steps)
    assert e["call"]["name"] == "calculate" and result == e["observation"]
    dependencies = [
        operation_evidence(x, steps[:j], prior) for j, x in enumerate(steps)
    ]
    val = Fraction(result["value"])
    pct = bool(re.search(r"percent|%", current["question"], re.I))
    # The publisher stores many percent changes as ratios. Full texts retain
    # the units for later review; allow the equivalent 100x calculation only
    # when the question explicitly asks for a percentage.
    tol = Fraction(51, 10**7)
    scale = (
        1
        if abs(val - expected) <= tol
        else 100
        if pct and abs(val - 100 * expected) <= 100 * tol
        else None
    )
    return dict(
        accepted=bool(
            scale
            and any(a for a, b in dependencies)
            and any(b for a, b in dependencies)
        ),
        calls=1,
        substantive=any(a for a, b in dependencies),
        previous_operand=any(b for a, b in dependencies),
        correct_scale=scale,
        returned=str(val),
        reference=str(expected),
    )


def run(freeze):
    for p in (DATA, ROOT / "inventory.json"):
        committed(p)
    cfg = read("research/experiments/cross-view-reading-01.json")["teacher"]
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    selected = select()
    order = []
    defs = declarations([PROVIDER])
    for w, i in selected:
        h, q = inputs(w, i)
        ids = tok.encode(render(tok, [*h, user(q)], defs), add_special_tokens=False)
        order.append(
            dict(
                id=w["id"],
                issuer=w["issuer"],
                turn=i,
                prompt=fingerprint(ids),
                tokens=len(ids),
            )
        )
    s = dict(
        material=digest(DATA),
        software=digest(Path(__file__)),
        teacher=cfg,
        order=order,
        protocol="Training source qualification only. On 24 different original publisher-training reports and issuers, ask an original follow-up which depends on an earlier arithmetic result. Supply the complete original report and earlier original user turns. Reconstruct historical assistant messages from published rounded quantities and actual preceding expert program executions. The training teacher is asked to use the real calculator. Give it no reference program, answer or gold evidence for the current question. Fixed ordinary-depth 4B Qwen teacher, official thinking, full-vocabulary greedy selection up to 512 tokens on one turn. Assess the next action only. Require exactly one actual successful substantive calculation using a previous real operand and returning the source value within 0.0000051, allowing equivalent percentage-point display for an explicitly percentage question. Require at least 12 accepted of 24 distinct issuers and all source/prompt and callback checks. Preserve all failures. Do not use this teacher on product evaluation prompts. A failing yield closes training from this collection. A passing yield permits a source review of the exact thinking and accepted actions before any independent student fit. No teacher in deployed model.",
        limits=dict(seconds=720, peak_gib=4.5, available_gib=1.5),
    )
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s and not (OUT / "receipt.json").exists()
    acquire()
    path = Path(cfg["directory"])
    assert digest(path / "model.safetensors") == cfg["weights_sha256"]
    assert all(digest(path / k) == v for k, v in cfg["files"].items())
    assert (
        Tokenizer.from_file(str(path / "tokenizer.json")).get_vocab() == tok.get_vocab()
    )
    mx.reset_peak_memory()
    start = monotonic()
    last = -math.inf

    def guard(_=0):
        nonlocal last
        now = monotonic()
        if now < last + 0.7:
            return
        last = now
        if (
            now - start > 720
            or mx.get_peak_memory() > 4.5 * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("training teacher resource limit")

    rows = []
    limit = None
    try:
        guard()
        model, _ = load_model(path)
        text = model.language_model
        net = LoopedModel(text, Span(len(text.layers) - 1, 1, 1))
        net.eval()
        for (w, i), expected in zip(selected, order, strict=True):
            h, q = inputs(w, i)
            tr = dialogue(
                net,
                tok,
                q,
                [PROVIDER],
                _messages=h,
                max_tokens=512,
                max_turns=1,
                tool_format="qwen",
                progress=guard,
                prefill_cache_state=True,
            )
            (t,) = tr["turns"]
            assert fingerprint(t["prompt_ids"]) == expected["prompt"]
            rows.append(
                dict(id=w["id"], turn=i, trajectory=tr, grade=assessment(w, i, tr))
            )
            packed(OUT / "traces.json.gz", rows)
            print("training teacher", len(rows), flush=True)
        last = -math.inf
        guard()
    except InterruptedError as e:
        limit = str(e)
    accepted = sum(r["grade"]["accepted"] for r in rows)
    persist(
        OUT / "receipt.json",
        dict(
            passed=not limit and len(rows) == 24 and accepted >= 12,
            limit=limit,
            plan=digest(PLAN),
            traces=digest(OUT / "traces.json.gz") if rows else None,
            completed=len(rows),
            accepted=accepted,
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
