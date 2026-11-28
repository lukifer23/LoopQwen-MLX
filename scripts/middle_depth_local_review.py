"""Review local responses without model labels and publish only workload-level aggregates."""

import argparse
from contextlib import ExitStack
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, save, digest
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_local_workflows import LOCAL, PLAN, frozen, providers


def records():
    cohort = frozen()
    cohort_hash = digest(LOCAL / "cohort.json")
    for case in cohort["cases"]:
        with ExitStack() as stack:
            lookup = {t.name: t for t in providers(case["kind"], LOCAL, stack)}
            for arm in ("base", "loop"):
                path = LOCAL / "traces" / arm / (case["id"] + ".json.gz")
                r = unpack(path)
                if r["arm"] != arm or r["cohort"] != cohort_hash:
                    raise ValueError("Local trajectory identity differs")
                history = []
                for i, (row, reference) in enumerate(
                    zip(r["rows"], case["refs"], strict=True)
                ):
                    trace = row["trace"]
                    if i == 0 and row["opening"] != case["opening"]:
                        raise ValueError("Different paired request")
                    if i and row["before"] != r["rows"][i - 1]["after"]:
                        raise ValueError("Missing own-history messages")
                    calls = []
                    for turn in trace["turns"]:
                        submitted = (
                            [turn["call"]] if "call" in turn else turn.get("calls", [])
                        )
                        observations = (
                            [turn["observation"]]
                            if "observation" in turn
                            else turn.get("observations", [])
                        )
                        if not observations:
                            continue
                        for call, obs in zip(submitted, observations, strict=True):
                            try:
                                replayed = lookup[call["name"]].callback(
                                    **call["arguments"]
                                )
                            except (
                                ValueError,
                                SyntaxError,
                                ArithmeticError,
                                TypeError,
                            ) as error:
                                replayed = {"error": type(error).__name__}
                            if replayed != obs:
                                raise ValueError(
                                    "Provider replay disagrees with the recorded local file"
                                )
                            calls.append(
                                {
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                    "observation": obs,
                                }
                            )
                    item = {
                        "prompt": case["prompts"][i],
                        "source_reference": reference,
                        "answer": trace["answer"],
                        "calls": calls,
                        "prior_calls": list(history),
                        "status": trace["status"],
                    }
                    yield arm, case, row, r, item
                    history.extend(calls)


def blind():
    deck = {}
    for _, _, _, _, item in records():
        if item["status"] == "answered":
            deck[fingerprint(item)] = item
    save(LOCAL / "review" / "blinded.json", {k: deck[k] for k in sorted(deck)})


def report():
    folder = LOCAL / "review"
    deck = read(folder / "blinded.json")
    grades = read(folder / "grades.json")
    if set(deck) != set(grades) or any(fingerprint(v) != k for k, v in deck.items()):
        raise ValueError("Complete the masked local review before comparison")
    rows = []
    for arm, case, row, raw, item in records():
        success = False
        if item["status"] == "answered":
            k = fingerprint(item)
            assert deck[k] == item
            success = grades[k]["correct"]
            assert type(success) is bool
        nonmath = case["kind"] != "math"
        if nonmath and success:
            evidence = item["calls"] + item["prior_calls"]
            assert any(
                c["name"] in ("database_query", "search_documents")
                and "error" not in c["observation"]
                for c in evidence
            )
        seconds = row["seconds"]
        rows.append(
            {
                "arm": arm,
                "case": case["id"],
                "kind": case["kind"],
                "correct": success,
                "by_deadline": success and seconds <= 30,
                "seconds": seconds,
                "capped_time_to_correct": min(seconds, 30) if success else 30,
                "first_visible": row["first_visible"],
                "peak": row["peak"],
                "status": item["status"],
                "load_seconds": raw["load_seconds"],
                "pid": raw["pid"],
                "block": case["block"],
                "calls": len(item["calls"]),
            }
        )
    save(folder / "scored.json", rows)
    closed = [
        read(p)
        for p in (ROOT / "charges").glob("*.json")
        if read(p)["job"].startswith("local-workflows/")
    ]
    arm_summaries = {}
    jobs = {}
    for arm in ("base", "loop"):
        local = [r for r in rows if r["arm"] == arm and r["kind"] != "math"]
        public = [r for r in rows if r["arm"] == arm and r["kind"] == "math"]
        jobs[arm] = {
            kind: all(r["correct"] for r in local if r["kind"] == kind)
            for kind in ("resume", "transfer")
        }
        assert len(local) == 4 and len(public) == 4
        own = [r for r in rows if r["arm"] == arm]
        pids = {r["pid"]: r["load_seconds"] for r in own}
        assert len(pids) == 2 and all(
            any(c["pid"] == p and c["status"] == "finished" for c in closed)
            for p in pids
        )
        visible = [r["first_visible"] for r in local if r["first_visible"] is not None]
        arm_summaries[arm] = {
            "complete_local_jobs": sum(jobs[arm].values()),
            "local_jobs": 2,
            "correct_local_requests": sum(r["correct"] for r in local),
            "local_requests": len(local),
            "local_correct_by_30_seconds": sum(r["by_deadline"] for r in local),
            "local_mean_capped_time_to_correct": fmean(
                r["capped_time_to_correct"] for r in local
            ),
            "local_elapsed": sum(r["seconds"] for r in local),
            "local_first_visible_mean": fmean(visible) if visible else None,
            "local_visible_requests": len(visible),
            "local_peak_bytes": max(r["peak"] for r in local),
            "replayed_callback_results": sum(r["calls"] for r in local),
            "gsm8k_correct": sum(r["correct"] for r in public),
            "gsm8k_questions": len(public),
            "gsm8k_elapsed": sum(r["seconds"] for r in public),
            "mean_process_load_seconds": fmean(pids.values()),
        }
    base, loop = arm_summaries["base"], arm_summaries["loop"]
    gained = sum(jobs["loop"][k] and not v for k, v in jobs["base"].items())
    lost = sum(v and not jobs["loop"][k] for k, v in jobs["base"].items())
    efficient = (
        base["complete_local_jobs"] == loop["complete_local_jobs"] == 2
        and loop["local_mean_capped_time_to_correct"]
        <= 0.8 * base["local_mean_capped_time_to_correct"]
    )
    save(
        ROOT / "local-workflow-screen.json",
        {
            "protocol": digest(PLAN),
            "arms": arm_summaries,
            "gained_jobs": gained,
            "lost_jobs": lost,
            "replication_signal": bool((gained and not lost) or efficient),
            "efficiency_condition": efficient,
            "worker_seconds": sum(c["seconds"] for c in closed),
            "scope": "Two project-local jobs with own-history follow-ups, and four separate public math items. Raw local files and review stay outside Git.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("blind", "report"))
    blind() if p.parse_args().action == "blind" else report()
