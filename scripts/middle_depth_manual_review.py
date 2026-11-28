"""Blind, score and audit the registered complete GNU manual pairs."""

import argparse
from math import comb
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.middle_depth_budget import BASE, ROOT
from scripts.middle_depth_manuals import FROZEN, STAGE, WORLD_ORDER, openings, restore
from scripts.middle_depth_pair_audit import serving
from scripts.middle_depth_review import first_visible

OUT = ROOT / STAGE


def review(score):
    frozen = read(FROZEN)
    tok = tokenizer({"checkpoint": str(BASE)})
    blinded, rows = {}, []
    for name, enrolled in frozen["manuals"].items():
        verified, (_, _, worlds, _) = openings(name, tok)
        if verified != enrolled:
            raise ValueError("Manual enrollment changed")
        for world in WORLD_ORDER:
            expected = enrolled["worlds"][world]
            for arm in ("base", "loop"):
                file = OUT / name / arm / (world + ".json")
                if not file.is_file():
                    continue
                receipt = read(file)
                archive = ROOT / receipt["archive"]
                if digest(archive) != receipt["archive_sha256"]:
                    raise ValueError("Trajectory archive changed")
                with TemporaryDirectory() as temp:
                    path = Path(temp) / "restored.json.gz"
                    restore(archive, path, *worlds[world])
                    if digest(path) != receipt["sha256"]:
                        raise ValueError("Raw trajectory changed")
                    raw = unpack(path)
                trace = raw["trace"]
                turns = trace["turns"]
                ids = turns[0]["prompt_ids"]
                if (
                    fingerprint(ids) != expected["input"]
                    or len(ids) != expected["length"]
                ):
                    raise ValueError("Input differs from registered paired start")
                calls = [
                    {
                        k: t[k]
                        for k in ("call", "calls", "observation", "observations")
                        if k in t
                    }
                    for t in turns
                    if "call" in t or "calls" in t
                ]
                item = {
                    "source": name,
                    "world": world,
                    "reference": expected["reference"],
                    "answer": trace["answer"],
                    "calls": calls,
                }
                key = fingerprint(item)
                if trace["status"] == "answered":
                    blinded[key] = item
                warm = trace["seconds"]
                prefill = raw["cold_build_seconds"]
                visible = first_visible(turns, raw["timing"], tok)
                rows.append(
                    {
                        "source": name,
                        "world": world,
                        "arm": arm,
                        "receipt": digest(file),
                        "status": trace["status"],
                        "review_key": key,
                        "input": expected["input"],
                        "input_tokens": len(ids),
                        "warm_seconds": warm,
                        "cold_seconds": warm + prefill,
                        "first_visible_cold": None
                        if visible is None
                        else prefill + visible,
                        "work": dict(serving(raw, arm)),
                        "peak_bytes": max(t["peak_bytes"] for t in turns),
                        "scratch_bytes": raw["scratch_bytes"],
                        "calls": calls,
                    }
                )
    if not score:
        save(OUT / "blinded.json", dict(sorted(blinded.items())))
        return
    if read(OUT / "blinded.json") != blinded:
        raise ValueError("Frozen masked responses differ")
    grades = read(OUT / "grades.json")
    if set(grades) != set(blinded):
        raise ValueError("Grade every completed visible response")
    for r in rows:
        g = (
            grades[r["review_key"]]
            if r["status"] == "answered"
            else {
                "recall": False,
                "complete": False,
                "unsupported": None,
                "reason": r["status"],
            }
        )
        if set(g) != {"complete", "recall", "unsupported", "reason"} or not g["reason"]:
            raise ValueError("Missing source-grounded grade")
        if g["complete"] and not g["recall"] or g["recall"] and g["unsupported"]:
            raise ValueError("Grade contradicts the source rubric")
        r["grade"] = g
        r["by_600"] = {
            k: g[k] and r["cold_seconds"] <= 600 for k in ("recall", "complete")
        }
        r["complete_capped_seconds"] = (
            r["cold_seconds"] if r["by_600"]["complete"] else 600
        )
    sources = {}
    for name in frozen["manuals"]:
        sources[name] = {}
        for arm in ("base", "loop"):
            panel = [r for r in rows if (r["source"], r["arm"]) == (name, arm)]
            both = {r["world"] for r in panel} == set(WORLD_ORDER)
            sources[name][arm] = {
                "measured": len(panel),
                "all_three_by_600": {
                    k: all(r["by_600"][k] for r in panel) if both else None
                    for k in ("recall", "complete")
                },
                "mean_complete_capped_seconds": mean(
                    r["complete_capped_seconds"] for r in panel
                )
                if panel
                else None,
            }
    decisions = {}
    for metric in ("recall", "complete"):
        fixed, lost, equal, missing = [], [], [], []
        for name, arms in sources.items():
            b, l = [arms[a]["all_three_by_600"][metric] for a in ("base", "loop")]
            if None in (b, l):
                missing.append(name)
            elif b == l:
                equal.append(name)
            else:
                (fixed if l else lost).append(name)
        discordant = len(fixed) + len(lost)
        decisions[metric] = {
            "gained": fixed,
            "lost": lost,
            "tied": equal,
            "incomplete": missing,
            "passed": bool(fixed) and not lost if not missing else None,
            "two_sided_exact_sign_p": min(
                1.0,
                2
                * sum(
                    comb(discordant, i) for i in range(min(len(fixed), len(lost)) + 1)
                )
                / 2**discordant,
            )
            if discordant
            else None,
        }
    save(
        OUT / "result.json",
        {
            "registration": digest(FROZEN),
            "blinded": digest(OUT / "blinded.json"),
            "grades": digest(OUT / "grades.json"),
            "rows": rows,
            "sources": sources,
            "decisions": decisions,
            "expected_requests": 12,
            "measured_requests": len(rows),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("blind", "score"))
    review(p.parse_args().action == "score")
