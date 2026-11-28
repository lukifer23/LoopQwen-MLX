"""Run native pass interventions within one frozen loop and audit answer transitions."""

import argparse
from pathlib import Path
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import FROZEN
from scripts.middle_depth_standalone import MODELS, check
from scripts.middle_depth_wave import dispatch

PLAN = Path("research/diagnostics/middle-depth/short-depth-probe.md")
WEIGHTS = "e8678f0d74d1eb6779dfea16d6f5677fe12021ad4fa4771317813a336b2a264a"


def run():
    if check(MODELS / "loop", "loop")["files"]["adapters.safetensors"] != WEIGHTS:
        raise ValueError("Incorrect pass intervention checkpoint")
    for block, passes in enumerate(((1, 3), (3, 1))):
        for depth in passes:
            dispatch(
                "development",
                "collect",
                "--stage",
                "depth",
                "--arm",
                "loop",
                "--policy",
                MODELS / "loop",
                "--depth",
                depth,
                "--tokens",
                512,
                "--block",
                block,
            )


def results():
    panel = {}
    for label, stage, arm, k in (
        ("base", "terminal", "base", 1),
        ("K1", "depth", "loop-k1", 1),
        ("K2", "terminal", "loop", 2),
        ("K3", "depth", "loop-k3", 3),
    ):
        rows = read(ROOT / "review" / stage / "scored.json")
        by = {}
        for row in rows:
            if row["cap"] != 512 or row["arm"] != arm:
                continue
            p = ROOT / row["file"]
            if digest(p) != row["sha"]:
                raise ValueError("Changed reviewed trajectory")
            raw = unpack(p)
            sites = [(i, 0) for i in range(24)]
            sites[16:16] = [(i, v) for v in range(1, k) for i in range(12, 16)]
            if list(map(tuple, raw["physical_sites"])) != sites or raw["weights"] != (
                "untouched-base" if label == "base" else WEIGHTS
            ):
                raise ValueError("Incorrect pass execution")
            attention = [s for s in sites if s[0] % 4 == 3]
            for t in raw["trace"]["turns"]:
                total = len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                if [
                    (h["layer"], h["turn"]) for h in t["kv_history"]
                ] != attention or any(
                    h["offset"] != total or h["retained"] != total
                    for h in t["kv_history"]
                ):
                    raise ValueError("Invalid physical attention histories")
            key = f"{row['source']}/{row['world']}"
            ids = fingerprint(raw["trace"]["turns"][0]["prompt_ids"])
            if row["world"] != "own" and ids != read(FROZEN)["ids"][key]:
                raise ValueError("Changed opening")
            ok = row["grade"]["correct"]
            by[key] = {
                "correct": ok,
                "family": row["family"],
                "by_deadline": ok and row["seconds"] <= 8,
                "capped": min(row["seconds"], 8) if ok else 8,
                "seconds": row["seconds"],
                "tokens": row["token_count"],
                "sha": row["sha"],
            }
        if len(by) != 31:
            raise ValueError("Incomplete native cohort")
        panel[label] = by
    primary = [
        key for key in panel["base"] if key.split("/")[1] not in ("own", "shared")
    ]
    summary = {}
    for label, by in panel.items():
        summary[label] = {
            "supported": sum(by[k]["correct"] for k in primary),
            "by_deadline": sum(by[k]["by_deadline"] for k in primary),
            "capped_mean": fmean(by[k]["capped"] for k in primary),
            "tokens": sum(by[k]["tokens"] for k in primary),
            "seconds": sum(by[k]["seconds"] for k in primary),
            "paired_hard_sources": sorted(
                {
                    s
                    for k, r in by.items()
                    if r["family"] in ("reasoning", "tools")
                    and (s := k.split("/")[0])
                    and by[s + "/original"]["by_deadline"]
                    and by[s + "/changed"]["by_deadline"]
                }
            ),
            "followups": {
                w: sum(v["correct"] for k, v in by.items() if k.endswith("/" + w))
                for w in ("own", "shared")
            },
        }
    changes = {}
    for a, b in (
        ("base", "K1"),
        ("base", "K2"),
        ("base", "K3"),
        ("K1", "K2"),
        ("K2", "K3"),
    ):
        changes[f"{a}-to-{b}"] = {
            "gained": [
                k
                for k in panel[a]
                if panel[b][k]["correct"] and not panel[a][k]["correct"]
            ],
            "lost": [
                k
                for k in panel[a]
                if panel[a][k]["correct"] and not panel[b][k]["correct"]
            ],
        }
        if a != "base":
            changes[f"{a}-to-{b}"]["replication_signal"] = (
                len(summary[b]["paired_hard_sources"])
                > len(summary[a]["paired_hard_sources"])
                and summary[b]["by_deadline"] >= summary[a]["by_deadline"]
            )
    save(
        ROOT / "direct" / "short-depth-probe.json",
        {
            "plan": digest(PLAN),
            "weights": WEIGHTS,
            "summary": summary,
            "transitions": changes,
            "rows": panel,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("run", "results"))
    run() if p.parse_args().stage == "run" else results()
