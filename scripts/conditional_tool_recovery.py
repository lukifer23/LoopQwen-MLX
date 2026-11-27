"""Measure what each deployed model did on the turn after an actual provider error."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv

P = Path("paper")
ROOT = Path("research/results/conditional-program-primary-01")
ARMS = ("candidate", "control")


def callback_turns(tr):
    return [
        [
            (
                e["call"]["name"],
                fingerprint(e["call"]["arguments"]),
                isinstance(e["observation"], dict) and "error" in e["observation"],
            )
            for e in executions({"turns": [t]})
        ]
        for t in tr["turns"]
    ]


def detail(record, arm, cap, success):
    (q,) = record["requests"]
    tr = q["trajectory"]
    turns = callback_turns(tr)
    calls = [e for t in turns for e in t]
    failed = next((i for i, t in enumerate(turns, 1) if any(e[2] for e in t)), None)
    after = [e for t in turns[failed:] for e in t] if failed else []
    first = [e for e in turns[failed - 1] if e[2]] if failed else []
    return {
        "id": record["id"],
        "family": record["family"],
        "arm": arm,
        "status": tr["status"],
        "grounded": success,
        "turns": len(turns),
        "max_turns": cap,
        "calls": len(calls),
        "successful_callbacks": sum(not e[2] for e in calls),
        "callback_errors": sum(e[2] for e in calls),
        "first_error_turn": failed,
        "could_generate_after_error": bool(failed and failed < cap),
        "generated_after_error": bool(failed and len(turns) > failed),
        "called_again": bool(after),
        "same_failed_arguments_again": any(
            u[:2] == v[:2] for u in after for v in first
        ),
        "changed_arguments_same_method": any(
            u[0] == v[0] and u[1] != v[1] for u in after for v in first
        ),
        "successful_later_callback": any(not e[2] for e in after),
        "successful_changed_arguments_same_method": any(
            not u[2] and u[0] == v[0] and u[1] != v[1] for u in after for v in first
        ),
        "grounded_after_error": bool(failed and success),
    }


def stats(rows):
    return {
        "requests": len(rows),
        "with_calls": sum(r["calls"] > 0 for r in rows),
        "with_returns": sum(r["successful_callbacks"] > 0 for r in rows),
        "calls": sum(r["calls"] for r in rows),
        "provider_errors": sum(r["callback_errors"] for r in rows),
        "requests_with_error": sum(bool(r["first_error_turn"]) for r in rows),
        "turns_first_error": dict(
            Counter(r["first_error_turn"] for r in rows if r["first_error_turn"])
        ),
        "status": dict(Counter(r["status"] for r in rows)),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                "could_generate_after_error",
                "generated_after_error",
                "called_again",
                "same_failed_arguments_again",
                "changed_arguments_same_method",
                "successful_later_callback",
                "successful_changed_arguments_same_method",
                "grounded_after_error",
            )
        },
    }


def summarize(root, streams, offered, grounded):
    reg = read(root / "registration.json")
    rows = []
    methods = defaultdict(Counter)
    for arm, stream in zip(ARMS, streams, strict=True):
        for r in stream:
            if r["id"] not in offered:
                continue
            (q,) = r["requests"]
            key = (r["id"], arm)
            success = grounded[key] if key in grounded else q["score"]["success"]
            rows.append(detail(r, arm, reg["max_turns"], success))
            for t in callback_turns(q["trajectory"]):
                for name, _, fail in t:
                    methods[arm, name]["error" if fail else "returned"] += 1
    assert len(rows) == 2 * len(offered)
    common = set.intersection(
        *[
            {
                r["id"]
                for r in rows
                if r["arm"] == arm and r["could_generate_after_error"]
            }
            for arm in ARMS
        ]
    )
    report = {
        "primary": digest(root / "checksums.json"),
        "common_opportunity_ids": sorted(common),
        "arms": {},
    }
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        report["arms"][arm] = {
            "all": stats(rs),
            "same_request_both_had_another_turn": stats(
                [r for r in rs if r["id"] in common]
            ),
            "by_family": {
                f: stats([r for r in rs if r["family"] == f])
                for f in sorted({r["family"] for r in rs})
            },
            "by_tool": {
                m: dict(c) for (a, m), c in sorted(methods.items()) if a == arm
            },
        }
    return report, rows


def build():
    with (P / "conditional-primary-requests.csv").open() as f:
        offered = {r["id"] for r in csv.DictReader(f) if r["tools_available"] == "True"}
    with (P / "conditional-primary-reading.csv").open() as f:
        grounded = {
            (r["id"], r["arm"]): r["grounded_usable"] == "True"
            for r in csv.DictReader(f)
            if r["family"] == "natural_mounted"
        }
    streams = [unpack(ROOT / (arm + ".json.gz")) for arm in ARMS]
    report, rows = summarize(ROOT, streams, offered, grounded)
    assert len(rows) == 356
    write_csv(P / "conditional-primary-tool-recovery.csv", rows)
    save(P / "conditional-primary-tool-recovery.json", report)
    return report


if __name__ == "__main__":
    for arm, r in build()["arms"].items():
        print(arm, r["all"], r["same_request_both_had_another_turn"])
