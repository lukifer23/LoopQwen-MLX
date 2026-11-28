"""Join source-locked direct-pair grades to real tool outcomes and native time."""

import argparse
import csv
from io import StringIO
from statistics import fmean
from collections import Counter

from loopqwen.io import read, save, digest
from loopqwen.identity import fingerprint
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_local_review import records, blind
from scripts.middle_depth_local_workflows import LOCAL
from scripts.middle_depth_specialist import WORK, PLAN, providers


def report():
    folder = WORK / "review"
    lock = read(ROOT / "specialist-review-lock.json")
    if (
        lock["deck"] != digest(folder / "blinded.json")
        or lock["grades"] != digest(folder / "grades.json")
        or lock["inputs"] != digest(ROOT / "specialist-inputs.json")
    ):
        raise ValueError("The masked response set, grades or inputs changed")
    deck = read(folder / "blinded.json")
    grades = read(folder / "grades.json")
    assert deck.keys() == grades.keys()
    scores = []
    for arm, case, row, raw, item in records(WORK, PLAN, providers):
        correct = False
        if item["status"] == "answered":
            k = fingerprint(item)
            assert deck[k] == item
            correct = grades[k]["correct"]
        ix = case["prompts"].index(item["prompt"])
        outcomes = [
            c
            for c in item["calls"]
            if c["name"] == "python" and c["observation"].get("exit_code") == 0
        ]
        if ix == 0 and case["kind"] == "files":
            correct = correct and bool(outcomes)
        if (
            ix == 1
            and case["kind"] in ("files", "supplied")
            and item["status"] == "answered"
        ):
            rows = list(csv.reader(StringIO(item["answer"].strip())))
            actual = bool(
                rows
                and rows[0] == ["family", "early_correct", "terminal_correct"]
                and sorted(rows[1:])
                == [["chat", "3", "4"], ["reasoning", "2", "1"], ["tools", "0", "0"]]
            )
            if actual != correct:
                raise ValueError("CSV parser and content judgment differ")
        scores.append(
            dict(
                arm=arm,
                case=case["id"],
                kind=case["kind"],
                index=ix,
                status=item["status"],
                correct=correct,
                seconds=row["seconds"],
                time_to_correct=min(30, row["seconds"]) if correct else 30,
                by_deadline=correct and row["seconds"] <= 30,
                first_visible=row["first_visible"],
                peak=row["peak"],
                python_successes=len(outcomes),
                callbacks=len(item["calls"]),
                pid=raw["pid"],
                load=raw["load_seconds"],
            )
        )
    assert len(scores) == 20
    save(folder / "scored.json", scores)
    original = read(LOCAL / "review/scored.json")
    charges = [
        r
        for f in (ROOT / "charges").glob("*.json")
        if (r := read(f))["job"].startswith("specialist-transfer/")
    ]
    assert len({r["pid"] for r in scores}) == 4 and {
        c["pid"] for c in charges if c["status"] == "finished"
    } == {r["pid"] for r in scores}
    arms = {}
    for arm in ("base", "loop"):
        own = [r for r in scores if r["arm"] == arm]
        math = [r for r in own if r["kind"] == "math"]
        prev = {
            r["case"]: r["correct"]
            for r in original
            if r["kind"] == "math" and r["arm"] == arm
        }
        assert prev.keys() == {r["case"] for r in math}
        parts = {}
        for kind in ("math", "meeting", "files", "supplied"):
            rs = [r for r in own if r["kind"] == kind]
            parts[kind] = dict(
                correct=sum(r["correct"] for r in rs),
                n=len(rs),
                jobs=sum(
                    all(r["correct"] for r in rs if r["case"] == case)
                    for case in {r["case"] for r in rs}
                ),
                statuses=dict(Counter(r["status"] for r in rs)),
                seconds=sum(r["seconds"] for r in rs),
                by_30s=sum(r["by_deadline"] for r in rs),
                mean_capped_time_to_correct=fmean(r["time_to_correct"] for r in rs),
                peak_bytes=max(r["peak"] for r in rs),
                executed_callbacks=sum(r["callbacks"] for r in rs),
                successful_python_processes=sum(r["python_successes"] for r in rs),
            )
        arms[arm] = {
            "groups": parts,
            "correct_math_pairs": sum(r["correct"] and prev[r["case"]] for r in math),
            "total_math_pairs": 4,
        }
    base, loop = arms["base"], arms["loop"]
    sentence = (
        f"On the four original and changed math pairs, LoopQwen solved both versions on {loop['correct_math_pairs']}/4 and Base on {base['correct_math_pairs']}/4. "
        f"On two complete QMSum transcripts, supported summaries were {loop['groups']['meeting']['correct']}/2 for LoopQwen and {base['groups']['meeting']['correct']}/2 for Base. "
        f"The local Python handoff and CSV follow-up yielded {loop['groups']['files']['jobs']}/1 complete jobs for LoopQwen and {base['groups']['files']['jobs']}/1 for Base. "
        f"With genuine stdout provided, the two models completed {loop['groups']['supplied']['jobs']}/1 and {base['groups']['supplied']['jobs']}/1 jobs, respectively."
    )
    save(
        ROOT / "specialist-result.json",
        {
            "arms": arms,
            "paper_summary": sentence,
            "inputs": lock["inputs"],
            "locked_review": digest(ROOT / "specialist-review-lock.json"),
            "worker_seconds": sum(c["seconds"] for c in charges),
            "scope": "Direct comparison of two frozen standalone models on small development sources. Private transcripts and signed-off review are retained locally.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("blind", "report"))
    if p.parse_args().action == "blind":
        blind(WORK, PLAN, providers)
    else:
        report()
