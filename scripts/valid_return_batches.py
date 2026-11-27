"""Account for proposed parallel calls, rejected batches and evidence visibility from raw model turns."""

from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save

ROOT = Path("research/results/valid-return-01")
OUT = Path("docs/measurements/valid-return-batches.json")


def run():
    panel = read(read(ROOT / "registration.json")["material"])
    results = {}
    for arm in ("candidate", "control"):
        totals = {"wheel": Counter(), "weekday": Counter()}
        cases = []
        for w, r in zip(panel["rows"], unpack(ROOT / (arm + ".json.gz")), strict=True):
            family = (
                "wheel"
                if w["family"] == "wheel"
                else "weekday"
                if w["id"].endswith("-weekday")
                else None
            )
            if not family:
                continue
            group = totals[family]
            tr = r["requests"][0]["trajectory"]
            t = tr["turns"][0]
            calls = t.get("calls", [t["call"]] if "call" in t else [])
            obs = t.get(
                "observations", [t["observation"]] if "observation" in t else []
            )
            got = {
                "first_batch": len(calls) > 1,
                "batch_rejected": len(calls) > 1 and not obs,
                "contains_expected_first_call": w["ordered_events"][0]["call"] in calls,
                "proposed_nonexistent_python_column": any(
                    c.get("arguments", {}).get("value_column") == "Python"
                    for c in calls
                ),
                "guessed_calendar_before_date_return": family == "weekday"
                and any(
                    c.get("name") in ("weekday_number", "day_of_year") for c in calls
                ),
                "all_turns_used_without_answer": tr["status"] == "turn_limit",
            }
            group.update(got)
            cases.append(
                {
                    "id": w["id"],
                    "family": family,
                    "proposed_opening": calls,
                    "opening_observations": obs,
                    "status": tr["status"],
                    **got,
                }
            )
        results[arm] = {
            "totals": {k: dict(v) for k, v in totals.items()},
            "cases": cases,
        }
    if OUT.exists():
        assert read(OUT) == results
    else:
        save(OUT, results)
    print({k: v["totals"] for k, v in results.items()})


if __name__ == "__main__":
    run()
