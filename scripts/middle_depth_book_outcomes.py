"""Verify the complete source, follow-up and abstention cohorts for the selected pair."""

from collections import Counter
from itertools import product
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_pair_audit import serving

SCORED = ROOT / "document-review/direct-primary-books/scored.json"


def summarize():
    rows = read(SCORED)
    expected = set(
        product(("anne", "book"), ("original", "edited", "irrelevant"), ("first",))
    )
    expected.update(
        product(("anne",), ("original", "edited"), ("own", "shared", "unanswerable"))
    )
    contracts = {}
    for cap, arm in product((128, 512), ("base", "loop")):
        panel = {}
        outcomes = {}
        subset = [r for r in rows if (r["cap"], r["arm"]) == (cap, arm)]
        for r in subset:
            k = (r["source"], r["world"], r["kind"])
            path = ROOT / r["file"]
            if k in panel or digest(path) != r["sha"]:
                raise ValueError("Duplicate or changed document trajectory")
            raw = unpack(path)
            tr = raw["trace"]
            counts = serving(raw, arm)
            if r["generated"] != counts["generated_tokens"] or r["input_tokens"] != len(
                tr["turns"][0]["prompt_ids"]
            ):
                raise ValueError("Mismatched physical input or output length")
            callbacks = sum(
                len(t.get("observations", [])) + int("observation" in t)
                for t in tr["turns"]
            )
            panel[k] = r
            outcomes["/".join(k)] = {
                "supported": r["grade"]["correct"],
                "error": r["grade"]["first_error"],
                "physical_input": r["input_tokens"],
                "prompt": r["input"],
                "cold_seconds": r["seconds_cold"],
                "warm_seconds": r["seconds_warm"],
                "cold_first_visible": r["first_visible_cold"],
                "actual_callbacks": callbacks,
                "generated": r["generated"],
                "sha": r["sha"],
            }
        if set(panel) != expected:
            raise ValueError("Incomplete primary book enrollment")
        groups = {}
        for kind in ("first", "own", "shared", "unanswerable"):
            part = [r for r in subset if r["kind"] == kind]
            groups[kind] = {
                "total": len(part),
                "supported": sum(r["grade"]["correct"] for r in part),
                "by_600s_cold": sum(r["cold_by_600"] for r in part),
                "stops": dict(Counter(r["status"] for r in part)),
                "mean_capped_cold_time_to_correct": fmean(
                    min(600, r["seconds_cold"]) if r["grade"]["correct"] else 600
                    for r in part
                ),
                "callbacks": sum(
                    v["actual_callbacks"]
                    for k, v in outcomes.items()
                    if k.endswith("/" + kind)
                ),
            }
        contracts[f"{cap}/{arm}"] = {"groups": groups, "rows": outcomes}
    for cap in (128, 512):
        base = contracts[f"{cap}/base"]["rows"]
        for key, r in contracts[f"{cap}/loop"]["rows"].items():
            if not key.endswith("/own") and r["prompt"] != base[key]["prompt"]:
                raise ValueError("Mismatched paired public prompt")
    save(
        ROOT / "direct/primary-books.json",
        {
            "source": digest(SCORED),
            "grades": digest(SCORED.with_name("grades.json")),
            "cases": len(rows),
            "contracts": contracts,
            "scope": "Two development novels. Original, changed and neutral openings; separate full-prefix own-history and shared-history calculator follow-ups and abstentions on Anne. Versions of the same novel are dependent. Cold times include measured prefix construction and warm generation.",
        },
    )


if __name__ == "__main__":
    summarize()
