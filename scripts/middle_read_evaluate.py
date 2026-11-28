"""Run the prespecified short tests in fresh processes and apply the source-level gate."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_callback_audit import audit
from scripts.middle_depth_development import FROZEN
from scripts.middle_depth_pair_audit import serving, source_uncertainty
from scripts.middle_depth_standalone import check
from scripts.middle_depth_wave import dispatch
from scripts.middle_read_bundle import DEST, REPORT
from scripts.middle_read_fit import EARLY

STAGE = "middle-read-early"


def collect():
    check(DEST, "loop", read(REPORT / "standalone.json"))
    for block, order in enumerate(((1, 2), (2, 1))):
        for depth in order:
            dispatch(
                "development",
                "collect",
                "--stage",
                STAGE,
                "--arm",
                "loop",
                "--policy",
                DEST,
                "--tokens",
                512,
                "--depth",
                depth,
                "--block",
                block,
            )


def result():
    scored = ROOT / "review" / STAGE / "scored.json"
    accepted = read(scored)
    frozen = read(FROZEN)["ids"]
    policy = read(DEST / "read-policy.json")
    expected = set(frozen) | {
        k.replace("/shared", "/own") for k in frozen if k.endswith("/shared")
    }
    rows = {}
    summary = {}
    for depth in (1, 2):
        arm = f"loop-k{depth}"
        panel = {}
        work = Counter()
        for r in accepted:
            if r["arm"] != arm:
                continue
            path = ROOT / r["file"]
            raw = unpack(path)
            key = f"{r['source']}/{r['world']}"
            if (
                key in panel
                or digest(path) != r["sha"]
                or raw["protocol"] != digest(FROZEN)
                or r["cap"] != 512
                or raw["weights"] != policy["weights"]
                or raw["parent"] != policy["parent"]
                or raw["read_policy"] != digest(DEST / "read-policy.json")
                or raw["additional_attention_reads"] != depth
                or raw["ordinary_layer_calls"] != 24
                or raw["physical_sites"] != [[i, 0] for i in range(24)]
            ):
                raise ValueError("Invalid standalone trajectory")
            turns = raw["trace"]["turns"]
            if (
                r["world"] != "own"
                and fingerprint(turns[0]["prompt_ids"]) != frozen[key]
            ):
                raise ValueError("Opening differs from registration")
            for t in turns:
                n = len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                if [
                    (h["layer"], h["turn"], h["retained"], h["offset"])
                    for h in t["kv_history"]
                ] != [(i, 0, n, n) for i in range(3, 24, 4)]:
                    raise ValueError("Missing ordinary attention history")
            visits = serving(raw, "ordinary")
            if r["world"] not in ("own", "shared"):
                work.update(visits)
            panel[key] = r
        if set(panel) != expected or len(panel) != 31:
            raise ValueError("Incomplete enrollment")
        rows[arm] = panel
        summary[arm] = {
            "opening_work": dict(
                work,
                extra_attention_token_reads=depth
                * work["warm_layer_token_visits"]
                // 24,
            ),
        }
    callback_path = REPORT / "callback-replay.json"
    audit((STAGE,), set(rows), callback_path)
    callbacks = read(callback_path)
    if callbacks["presentations"] != 26:
        raise ValueError("Incomplete tool action audit")
    baseline = read(ROOT / "direct/short-depth-probe.json")
    for arm, panel in rows.items():
        openings = {
            k: v for k, v in panel.items() if v["world"] not in ("own", "shared")
        }
        qualified = {
            k
            for k, v in openings.items()
            if v["grade"]["correct"] and v["seconds"] <= 8
        }
        sources = sorted(
            {
                v["source"]
                for v in openings.values()
                if v["family"] in ("reasoning", "tools")
                and v["source"] + "/original" in qualified
                and v["source"] + "/changed" in qualified
            }
        )
        numeric = [
            r
            for r in callbacks["rows"]
            if r["arm"] == arm
            and r["source"] in ("ink", "tiles")
            and r["world"] in ("original", "changed", "irrelevant")
        ]
        summary[arm] |= {
            "numeric_openings": len(numeric),
            "first_calculator_returned_value": sum(
                bool(r["replayed"]) and "value" in r["replayed"][0] for r in numeric
            ),
            "first_calculator_returned_target": sum(
                bool(r["replayed"]) and r["replayed"][0].get("value") == r["target"]
                for r in numeric
            ),
            "any_calculator_returned_target": sum(
                r["target_observed"] for r in numeric
            ),
            "supported": sum(v["grade"]["correct"] for v in openings.values()),
            "by_deadline": len(qualified),
            "paired_hard_sources": sources,
            "seconds": sum(v["seconds"] for v in openings.values()),
            "mean_capped_time_to_correct": fmean(
                min(8, v["seconds"]) if v["grade"]["correct"] else 8
                for v in openings.values()
            ),
            "followups": {
                w: sum(v["grade"]["correct"] for v in panel.values() if v["world"] == w)
                for w in ("own", "shared")
            },
            "by_source": {k: v["grade"]["correct"] for k, v in panel.items()},
        }
    one, two = (summary[f"loop-k{d}"] for d in (1, 2))
    parent = baseline["summary"]["K2"]
    checks = {
        "additional_qualifying_source": len(two["paired_hard_sources"])
        > len(one["paired_hard_sources"]),
        "opening_retention": two["supported"]
        >= max(one["supported"], parent["supported"]),
    }
    differences = {}
    for name, reference in (
        ("one_to_two", one["by_source"]),
        ("base_to_two", {k: v["correct"] for k, v in baseline["rows"]["base"].items()}),
    ):
        groups = defaultdict(dict)
        for k, r in rows["loop-k2"].items():
            if r["world"] in ("own", "shared"):
                continue
            family = groups[r["family"]]
            src = r["source"]
            family[src] = family.get(src, Fraction()) + Fraction(
                int(two["by_source"][k]) - int(reference[k]), 3
            )
        differences[name] = source_uncertainty(groups)
    save(
        REPORT / "early-decision.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "stage": STAGE,
            "registration": digest(EARLY),
            "rows": digest(scored),
            "callback_replay": digest(callback_path),
            "standalone": digest(REPORT / "standalone.json"),
            "summary": summary,
            "short_source_uncertainty": differences,
            "base": baseline["summary"]["base"],
            "parent": parent,
            "gains": [
                k for k, v in two["by_source"].items() if v and not one["by_source"][k]
            ],
            "losses": [
                k for k, v in one["by_source"].items() if v and not two["by_source"][k]
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "result"))
    if parser.parse_args().action == "collect":
        collect()
    else:
        result()
