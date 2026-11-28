"""Compare complete paired answers under the registered 512 and 4,096 output allowances."""

import argparse
from collections import Counter
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import SOURCE
from scripts.middle_depth_headroom import REG
from scripts.middle_depth_pair_audit import serving


def report(long):
    chosen = read(ROOT / "direct-decision.json")
    if long:
        plan_path = ROOT / "headroom/direct-complete-books.json"
        plan = read(plan_path)
        enrolled = {
            (src, world, kind)
            for key in plan["reasons"]
            for src, world in [key.split("/")]
            for kind in (
                ("first", "own", "shared", "unanswerable")
                if src == "anne" and world != "irrelevant"
                else ("first",)
            )
        }
        directory = ROOT / "document-review"
        reviews = {512: "direct-primary-books", 4096: "direct-final-books"}
        deadline = 600
    else:
        plan_path = ROOT / "headroom/direct-headroom.json"
        plan = read(plan_path)
        enrolled = {(k.split("/")[0], k.split("/")[1], "first") for k in plan["items"]}
        for r in read(SOURCE):
            if "followup" in r and r["id"] + "/original" in plan["items"]:
                enrolled.update((r["id"], "original", w) for w in ("own", "shared"))
        directory = ROOT / "review"
        reviews = {512: chosen["short_stage"], 4096: "direct-headroom"}
        deadline = 8
    if plan["registration"] != digest(REG):
        raise ValueError("Headroom registration differs")
    cells = {}
    grade_sources = {}
    for cap, stage in reviews.items():
        scored = directory / stage / "scored.json"
        grade_sources[str(cap)] = digest(scored)
        records = read(scored)
        for r in records:
            arm = r["arm"]
            if arm not in ("base", "loop") or r["cap"] != cap:
                continue
            kind = (
                r["kind"]
                if long
                else (r["world"] if r["world"] in ("own", "shared") else "first")
            )
            world = "original" if not long and kind in ("own", "shared") else r["world"]
            key = (r["source"], world, kind)
            if key not in enrolled:
                continue
            path = ROOT / r["file"]
            if digest(path) != r["sha"]:
                raise ValueError("Changed scored response")
            if cap == 512:
                prior = (
                    plan["sources"][r["file"]]
                    if long
                    else plan["source_trajectories"][
                        r["source"] + "/" + r["world"] + "/" + arm
                    ]
                )
                if prior != r["sha"]:
                    raise ValueError("Earlier response differs from enrollment")
            raw = unpack(path)
            correct = r["grade"]["correct"]
            elapsed = r["seconds_cold"] if long else r["seconds"]
            identity = (arm, *key, cap)
            if identity in cells:
                raise ValueError("Duplicate headroom row")
            cells[identity] = {
                "file": r["file"],
                "sha": r["sha"],
                "correct": correct,
                "seconds": elapsed,
                "by_deadline": correct and elapsed <= deadline,
                "capped_time": min(elapsed, deadline) if correct else deadline,
                "status": r["status"],
                "error": r["grade"]["first_error"],
                "first_visible": r["first_visible_cold"]
                if long
                else r["first_visible"],
                "peak": r["peak"],
                "work": dict(serving(raw, arm)),
                "input": fingerprint(raw["trace"]["turns"][0]["prompt_ids"]),
            }
    expected = {
        (a, *k, c) for a in ("base", "loop") for k in enrolled for c in (512, 4096)
    }
    if set(cells) != expected:
        raise ValueError(f"Incomplete pairs: {sorted(expected - set(cells))}")
    outputs = {}
    for kind in sorted({k[-1] for k in enrolled}):
        for arm in ("base", "loop"):
            pairs = {}
            for k in sorted(enrolled):
                if k[-1] != kind:
                    continue
                old, new = (cells[arm, *k, c] for c in (512, 4096))
                if kind != "own" and old["input"] != new["input"]:
                    raise ValueError("Uncontrolled prompt change across output caps")
                other = cells["base" if arm == "loop" else "loop", *k, 4096]
                if kind != "own" and new["input"] != other["input"]:
                    raise ValueError("Unmatched opening across models")
                pairs["/".join(k)] = {"512": old, "4096": new}
            summaries = {}
            for c in ("512", "4096"):
                rows = [v[c] for v in pairs.values()]
                summaries[c] = {
                    "supported": sum(x["correct"] for x in rows),
                    "by_deadline": sum(x["by_deadline"] for x in rows),
                    "mean_capped": fmean(x["capped_time"] for x in rows),
                    "seconds": sum(x["seconds"] for x in rows),
                    "stops": dict(Counter(x["status"] for x in rows)),
                }
            outputs[f"{arm}/{kind}"] = {
                "pairs": pairs,
                "n": len(pairs),
                "summary": summaries,
                "gains": [
                    k
                    for k, p in pairs.items()
                    if p["4096"]["correct"] and not p["512"]["correct"]
                ],
                "losses": [
                    k
                    for k, p in pairs.items()
                    if p["512"]["correct"] and not p["4096"]["correct"]
                ],
            }
    save(
        ROOT
        / "headroom"
        / ("document-outcomes.json" if long else "short-outcomes.json"),
        {
            "enrollment": digest(plan_path),
            "scored": grade_sources,
            "deadline_seconds": deadline,
            "output_caps": [512, 4096],
            "groups": outputs,
            "scope": "Mandatory originals and cases selected by earlier token-limit status. Own-history and shared-history outcomes remain separate.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", choices=("short", "documents"))
    report(p.parse_args().panel == "documents")
