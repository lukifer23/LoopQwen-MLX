"""Track which real requests became usable and which earlier successes were lost."""

import argparse
from collections import Counter
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT


def requests(stage, arm, cap):
    rows = read(ROOT / "review" / stage / "scored.json")
    result = {}
    for row in rows:
        if row["arm"] != arm or row["cap"] != cap:
            continue
        path = ROOT / row["file"]
        if digest(path) != row["sha"]:
            raise ValueError("Reviewed trajectory changed")
        raw = unpack(path)
        observations = []
        turns = raw["trace"]["turns"]
        for t in turns:
            if "observation" in t:
                observations.append(t["observation"])
            observations.extend(t.get("observations", []))
        key = (row["source"], row["world"])
        result[key] = {
            "initial": raw["initial"],
            "family": row["family"],
            "correct": row["grade"]["correct"],
            "first_error": row["grade"]["first_error"],
            "status": row["status"],
            "completed": row["status"] == "answered",
            "seconds": row["seconds"],
            "callbacks": len(observations),
            "successful_callbacks": sum("error" not in v for v in observations),
            "evidence": row["file"],
            "sha": row["sha"],
        }
    return result


def measure(stage, arm, baseline_stage, baseline_arm, cap):
    current = requests(stage, arm, cap)
    baseline = requests(baseline_stage, baseline_arm, cap)
    if not current or current.keys() != baseline.keys():
        raise ValueError("Both sides require the same complete cohort")
    rows = []
    for key in sorted(current):
        source, world = key
        a, b = current[key], baseline[key]
        if world != "own" and a["initial"] != b["initial"]:
            raise ValueError(f"Paired model inputs differ on {key}")
        rows.append(
            {
                "source": source,
                "world": world,
                "family": a["family"],
                "current": a,
                "baseline": b,
                "delta": int(a["correct"]) - int(b["correct"]),
            }
        )
    summary = {}
    for family in sorted({r["family"] for r in rows}):
        sources = sorted({r["source"] for r in rows if r["family"] == family})
        report = {}
        for source in sources:
            worlds = {r["world"]: r for r in rows if r["source"] == source}
            if not {"original", "changed", "irrelevant"}.issubset(worlds):
                raise ValueError("Source missing a registered presentation")
            pairs = {}
            for side in ("current", "baseline"):
                orig = worlds["original"][side]["correct"]
                pairs[side] = {
                    "changes_correctly": orig and worlds["changed"][side]["correct"],
                    "stays_correct": orig and worlds["irrelevant"][side]["correct"],
                }
            report[source] = pairs
        cell = [
            r
            for r in rows
            if r["family"] == family
            and r["world"] in ("original", "changed", "irrelevant")
        ]
        summary[family] = {
            "sources": report,
            "presentations": len(cell),
            "gains": sum(r["delta"] == 1 for r in cell),
            "losses": sum(r["delta"] == -1 for r in cell),
            "outcomes": {
                side: {
                    "success": sum(r[side]["correct"] for r in cell),
                    "completed": sum(r[side]["completed"] for r in cell),
                    "requests_with_successful_callbacks": sum(
                        r[side]["successful_callbacks"] > 0 for r in cell
                    ),
                    "requests_with_any_callback": sum(
                        r[side]["callbacks"] > 0 for r in cell
                    ),
                    "stops": dict(Counter(r[side]["status"] for r in cell)),
                    "seconds": sum(r[side]["seconds"] for r in cell),
                }
                for side in ("current", "baseline")
            },
        }
    save(
        ROOT
        / "progress"
        / f"{stage}-{arm}-from-{baseline_stage}-{baseline_arm}-{cap}.json",
        {
            "summary": summary,
            "rows": rows,
            "macro_delta": fmean(
                (r["gains"] - r["losses"]) / r["presentations"]
                for r in summary.values()
            ),
            "definition": "Opening score counts complete supported answers. A changed pair requires success in both source worlds. A stability pair requires success in original and irrelevant worlds. Own/shared are reported separately. A provider result without an error is not proof of relevant operands or successful delivery.",
        },
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("stage", "arm", "baseline_stage", "baseline_arm"):
        p.add_argument(k)
    p.add_argument("cap", type=int)
    a = p.parse_args()
    for k, v in measure(**vars(a)).items():
        print(
            k,
            v["gains"],
            "gained,",
            v["losses"],
            "lost,",
            v["outcomes"]["current"]["success"],
            "supported",
        )
