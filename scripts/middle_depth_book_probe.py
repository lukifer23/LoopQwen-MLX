"""Identify whether the second native pass supplies changed-age recall on Anne."""

import argparse
from pathlib import Path
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_documents import FROZEN
from scripts.middle_depth_short_probe import WEIGHTS
from scripts.middle_depth_standalone import MODELS, check
from scripts.middle_depth_wave import dispatch

PLAN = Path("research/diagnostics/middle-depth/book-recall-depth.md")
STAGE = "recall-depth"
REVIEW = ROOT / "document-review" / STAGE


def run():
    manifest = check(MODELS / "loop", "loop")
    selected = read(ROOT / "direct-decision.json")
    short = read(ROOT / "direct/short-depth-probe.json")
    if (
        manifest["files"]["adapters.safetensors"] != WEIGHTS
        or selected["models"]["loop"]["weights"] != WEIGHTS
        or short["weights"] != WEIGHTS
    ):
        raise ValueError("Depth intervention weights differ from the selected loop")
    for world in ("edited", "original"):
        dispatch(
            "documents",
            "run",
            "--arm",
            "loop",
            "--policy",
            MODELS / "loop",
            "--depth",
            1,
            "--stage",
            STAGE,
            "--book",
            "anne",
            "--world",
            world,
            "--tokens",
            512,
            "--cache-mode",
            "resident",
        )


def results():
    component = read(REVIEW / "fact-grades.json")
    study = read(FROZEN)["anne"]
    historical = read(ROOT / "parent-evidence-recall.json")["review"]
    rows = {}
    for label, arm, folder, weights, depth in (
        ("base", "base", "parent", "base-dc7cdfe", 1),
        ("K2", "loop", "parent", WEIGHTS, 2),
        ("K1", "loop-k1", STAGE, WEIGHTS, 1),
    ):
        source = ROOT / "document-review" / folder
        grades = read(source / "scored.json")
        facts = component if depth == 1 and label == "K1" else historical
        pair = {}
        for r in grades:
            if (r["arm"], r["source"], r["cap"], r["kind"]) != (
                arm,
                "anne",
                512,
                "first",
            ):
                continue
            world = r["world"]
            if world not in ("original", "edited"):
                continue
            path = ROOT / r["file"]
            if digest(path) != r["sha"] or world in pair:
                raise ValueError("Changed or repeated trajectory")
            raw = unpack(path)
            sites = [(i, 0) for i in range(24)]
            if depth == 2:
                sites[16:16] = [(i, 1) for i in range(12, 16)]
            if (
                list(map(tuple, raw["sites"])) != sites
                or raw["weights"] != weights
                or raw["source"] != study["source"]
                or raw["frozen"] != digest(FROZEN)
            ):
                raise ValueError("Wrong execution or source")
            opening = raw["trace"]["turns"][0]["prompt_ids"]
            if {"input": fingerprint(opening), "length": len(opening)} != study[
                "worlds"
            ][world]:
                raise ValueError("Opening differs from the enrolled full edition")
            histories = [(i, k) for i, k in sites if i % 4 == 3]
            for turn in raw["trace"]["turns"]:
                last = len(turn["prompt_ids"]) + len(turn["token_ids"]) - 1
                kv = turn["kv_history"]
                if [(x["layer"], x["turn"]) for x in kv] != histories or any(
                    x["offset"] != last or x["retained"] != last for x in kv
                ):
                    raise ValueError("Missing full-attention state on a physical visit")
            correct = r["grade"]["correct"]
            fact = (
                r["status"] == "answered"
                and facts[r["review_key"]]["all_required_facts"]
            )
            cold = r["seconds_cold"]
            pair[world] = {
                "trace": r["file"],
                "sha": r["sha"],
                "review_key": r["review_key"],
                "length": len(opening),
                "cold_seconds": cold,
                "status": r["status"],
                "facts": fact,
                "correct": correct,
                "facts_by_600": fact and cold <= 600,
                "complete_by_600": correct and cold <= 600,
                "peak_bytes": r["peak"],
                "scratch_bytes": r["scratch_bytes"],
                "physical_visits": len(sites),
            }
        if set(pair) != {"original", "edited"}:
            raise ValueError("Missing edition")
        rows[label] = pair
    summary = {
        arm: {
            "fact_pair": all(x["facts_by_600"] for x in pair.values()),
            "correct": sum(x["correct"] for x in pair.values()),
            "capped_time_to_correct": fmean(
                min(x["cold_seconds"], 600) if x["correct"] else 600
                for x in pair.values()
            ),
        }
        for arm, pair in rows.items()
    }
    save(
        ROOT / "direct/book-recall-depth.json",
        {
            "plan": digest(PLAN),
            "one_pass_component_review": digest(REVIEW / "fact-grades.json"),
            "original_component_review": digest(ROOT / "parent-evidence-recall.json"),
            "summary": summary,
            "rows": rows,
            "local_second_pass_recall_benefit": summary["K2"]["fact_pair"]
            and not summary["K1"]["fact_pair"],
        },
    )


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("stage", choices=("run", "results"))
    run() if arg.parse_args().stage == "run" else results()
