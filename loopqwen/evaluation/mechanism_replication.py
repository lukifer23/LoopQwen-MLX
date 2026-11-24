"""Prospectively gated independent seeds and a fresh, map-disjoint confirmation."""

import random
from pathlib import Path

from ..data import pointer5, pointer6
from ..identity import fingerprint
from ..io import read, save
from ..training.mechanism import StudyArm, train
from .mechanism import evaluate
from .order_scores import metrics, paired


def fresh_maps():
    # Sealed historical rows contribute only exclusion IDs.
    blocked = {
        r["map_id"]
        for split in ("train", "development", "confirmation")
        for r in pointer5.rows(split)
    }
    for case in pointer5.rows("cases"):
        blocked.update(
            v["map_id"] for v in case.values() if isinstance(v, dict) and "map_id" in v
        )
    rng = random.Random(671124)
    result = []
    while len(result) < 384:
        r = pointer5.instance(rng.sample(range(8), 8), rng.randrange(8))
        if r["map_id"] in blocked or pointer5.cycle(r) <= 3:
            continue
        blocked.add(r["map_id"])
        result.append(r)
    return result


def replicate():
    spec = pointer6.SPEC
    root = Path("outputs") / spec["id"]
    decision = read(root / str(spec["seed"]) / "decision.json")
    if not decision["promoted"]:
        raise ValueError("Primary robust-generalization gate did not pass")
    control = decision["strongest_dense"]
    cohort = pointer5.rows("development")
    orders = ["canonical", *[k for k in spec["orders"] if k.startswith("random")]]
    config = read("experiments/phase6_followup.json")
    fresh = fresh_maps()
    manifest = {
        "registration": fingerprint(config),
        "ordered_map_stream": fingerprint(fresh),
        "maps": len(fresh),
    }
    dest = pointer6.ROOT / "fresh-confirmation.json"
    if dest.exists() and read(dest) != manifest:
        raise ValueError("Confirmation identity changed")
    save(dest, manifest)
    result = {
        "control": control,
        "development": {},
        "confirmation": {},
        "manifest": manifest,
    }
    for seed in spec["replication_seeds"]:
        comparison = {}
        for name in [control, "randomized"]:
            a = StudyArm.registered(name, seed)
            train(a)
            raw = evaluate(a, 480, cohort, orders, 3, a.output / "replication.json.gz")
            comparison[name] = metrics(raw["records"])
        dense = comparison[control]
        rec = comparison["randomized"]
        c = rec["conditions"]
        passed = (
            min(c["canonical"]["accuracy"]) >= spec["minimum_canonical"]
            and min(c["mean_random"]["accuracy"]) >= spec["minimum_random"]
            and c["mean_random"]["trajectory"][-1] >= spec["minimum_random_trajectory"]
            and c["mean_random"]["accuracy"][-1]
            > dense["conditions"]["mean_random"]["accuracy"][-1]
        )
        result["development"][seed] = {
            "scores": comparison,
            "paired": paired(dense, rec),
            "passed": passed,
        }
        save(root / "replication.json", result)
    if all(v["passed"] for v in result["development"].values()):
        for seed in [spec["seed"], *spec["replication_seeds"]]:
            comparison = {}
            for name in [control, "randomized"]:
                a = StudyArm.registered(name, seed)
                raw = evaluate(
                    a, 480, fresh, orders, 3, a.output / "confirmation.json.gz"
                )
                comparison[name] = metrics(raw["records"])
            result["confirmation"][seed] = {
                "scores": comparison,
                "paired": paired(comparison[control], comparison["randomized"]),
            }
            save(root / "replication.json", result)
