"""Enroll actual deliberation prefixes of the fitted recurrent policy on training sources."""

import argparse
from collections import Counter
from pathlib import Path
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen import qwen_tools
from loopqwen.evaluation.generation_eval import generate
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.native_function_data import token_pair
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.register_repair_query import FIT, RESULT, ROOT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

DATA = ROOT / "verified-training.json.gz"
PLAN = Path("research/diagnostics/learner-prefix-material.json")
DIR = Path("research/sources/learner-prefix")
RECORDS = DIR / "native-prefixes.json.gz"
RECEIPT = DIR / "qualification.json"
COUNTS = dict(
    strategy=192,
    report_action=160,
    report_reply=96,
    repair_known_schema=64,
    repair_after_schema=48,
    redirect_valid_read=64,
    answer_valid_read=48,
)
SEED = 937501


def selected():
    rows = unpack(DATA)
    pool = []
    for group, count in COUNTS.items():
        ranked = sorted(
            (r for r in rows if r["group"] == group),
            key=lambda r: fingerprint([SEED, "source", r["id"]]),
        )
        assert len(ranked) >= count
        pool += ranked[:count]
    return sorted(pool, key=lambda r: fingerprint([SEED, "order", r["id"]]))


def plan(rows):
    return dict(
        sources={
            str(p): digest(p)
            for p in (
                DATA,
                RESULT / FIT / "checksums.json",
                Path("docs/measurements/staged-query-reading.json"),
                RESULT / "staged-query-transfer-01" / "checksums.json",
            )
        },
        implementation=digest(Path(__file__)),
        seed=SEED,
        groups=COUNTS,
        ordered=[r["id"] for r in rows],
        question="Can the trained two-read policy supply its own genuine intermediate deliberation states on original training sources for subsequent correction learning?",
        protocol="Use only previously enrolled training sources. Run native two-read greedy decoding with original reasoning-enabled prompt, declarations and weights. Cycle 24, 64 and 112 predicted positions across the frozen order, stopping before a close-thought or end-of-turn token. Retain every emitted ID, including the stop token when reached. Each row must reproduce its exact original opening IDs and use two physical recurrence visits per predicted position. Require at least 64 strategy and 48 financial action cases reaching their assigned prefix length without closing reasoning. No inference-time token forcing or edits. If coverage, physical identity or resource budget fails, close the dependent fit.",
        limits=dict(seconds=1800, peak_gib=3, available_gib=1.5),
        basis="https://proceedings.mlr.press/v15/ross11a.html",
    )


def collect(rows, spec):
    import subprocess

    committed(PLAN)
    assert read(PLAN) == spec
    for p in spec["sources"]:
        committed(Path(p))
    if RECORDS.exists() or RECEIPT.exists():
        raise FileExistsError("The material already has a published record")
    before = monotonic()
    peak = 0
    last_check = 0

    def check(_=None):
        nonlocal peak, last_check
        now = monotonic()
        if now - last_check < 1:
            return
        last_check = now
        peak = max(peak, mx.get_peak_memory())
        if (
            now - before > spec["limits"]["seconds"]
            or peak > spec["limits"]["peak_gib"] * 2**30
            or psutil.virtual_memory().available
            < spec["limits"]["available_gib"] * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise RuntimeError("Native data-collection resource limit")

    check()
    actor = SharedQueryAssistant("exports/tool-language-packed-05", RESULT / FIT)
    net, tok = actor._tool_model, actor.tokenizer
    close = tok.convert_tokens_to_ids("</think>")
    assert tok.decode([close]) == "</think>"
    stops = {close, *tok.eos_token_ids}
    raw = []
    counts = Counter()
    try:
        for i, r in enumerate(rows):
            check()
            opening, _ = token_pair(tok, r)
            prompt = qwen_tools.render(tok, r["messages"], r["tools"])
            limit = (24, 64, 112)[i % 3]
            visits = net.output_revisits
            result = generate(
                net,
                tok,
                prompt,
                max_tokens=limit,
                stop_ids=[close],
                progress=check,
                return_prompt_ids=True,
                prefill_cache_state=True,
            )
            ids = result["token_ids"]
            assert (
                result["prompt_ids"] == opening
                and visits + 2 * len(ids) == net.output_revisits
            )
            assert not stops.intersection(ids[:-1])
            closed = ids[-1] in stops
            counts[r["group"]] += not closed
            raw.append(
                dict(
                    id=r["id"],
                    group=r["group"],
                    opening=fingerprint(opening),
                    tokens=ids,
                    limit=limit,
                    closed=closed,
                    visits=net.output_revisits - visits,
                    seconds=result["elapsed_s"],
                )
            )
            peak = max(peak, result["peak_bytes"])
            if len(raw) % 50 == 0:
                print(len(raw), flush=True)
        check()
    finally:
        DIR.mkdir(parents=True, exist_ok=True)
        packed(RECORDS, raw)
        persist(
            RECEIPT,
            dict(
                plan=digest(PLAN),
                sha256=digest(RECORDS),
                rows=len(raw),
                reached_limit=dict(counts),
                seconds=monotonic() - before,
                peak_bytes=peak,
                passed=len(raw) == len(rows)
                and counts["strategy"] >= 64
                and counts["report_action"] >= 48,
            ),
        )
    print(read(RECEIPT), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["register", "collect"])
    args = p.parse_args()
    rows = selected()
    spec = plan(rows)
    if args.action == "register":
        persist(PLAN, spec)
    else:
        collect(rows, spec)
