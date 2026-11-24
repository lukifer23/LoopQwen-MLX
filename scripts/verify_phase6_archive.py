"""Rescore archived state trajectories, paired orders and transplants using the standard library."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(p):
    data = p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix == ".gz" else data)


def score(result):
    counts = defaultdict(Counter)
    reference = {}
    members = defaultdict(set)
    same = defaultdict(Counter)
    for r in result["records"]:
        mapping = r["map"]
        start = r["start"]
        n = len(mapping)
        if (
            min(mapping) < 0
            or max(mapping) >= n
            or sorted(r["order"]) != list(range(n))
        ):
            raise ValueError("Malformed mapping/order")
        if hashlib.sha256(bytes(mapping)).hexdigest() != r["map_id"]:
            raise ValueError("Incorrect map ID")
        key = (r["map_id"], start)
        condition = r["condition"]
        if key in members[condition]:
            raise ValueError("Duplicate request")
        members[condition].add(key)
        node = start
        pred = []
        all_right = True
        for j, (gold, p) in enumerate(
            zip(r["gold"], r["probabilities"], strict=True), 1
        ):
            node = mapping[node]
            if (
                node != gold
                or len(p) != n
                or abs(sum(p) - 1) > 1e-5
                or any(not math.isfinite(v) or v < 0 for v in p)
            ):
                raise ValueError("Invalid gold/probabilities")
            pred.append(max(range(n), key=p.__getitem__))
            correct = pred[-1] == node
            all_right &= correct
            counts[condition][f"hop{j}"] += correct
            counts[condition][f"path{j}"] += all_right
            if condition != "canonical":
                base = reference[key][j - 1]
                same[condition][f"hop{j}"] += pred[-1] == base
                same[condition][f"both{j}"] += correct and base == node
        if condition == "canonical":
            reference[key] = pred
    if any(ids != set(reference) for ids in members.values()):
        raise ValueError("Mismatched map clusters")
    return counts, same, len(reference)


def patches(result):
    for test in result["interventions"].values():
        for key, value in test["summary"].items():
            kind, subset = key.split("/")
            n = target = natural = noise = recipient = donor = 0
            for r in test["records"]:
                d = r["donor"]
                a = r["recipient"]
                if kind != r["kind"]:
                    continue
                state = d["map"][d["start"]]
                if subset == "correct_donor" and r["donor_read"] != state:
                    continue
                expected = a["map"][state]
                own = a["map"][a["map"][a["start"]]]
                if expected != r["expected"] or own != r["recipient_next"]:
                    raise ValueError("Invalid intervention target")
                n += 1
                target += r["patch"] == expected
                natural += r["natural"] == expected
                noise += r["noise"] == expected
                recipient += r["patch"] == own
                donor += r["patch"] == d["map"][state]
            if [n, target, natural, noise, recipient, donor] != [
                value[k]
                for k in (
                    "n",
                    "donor_transition",
                    "natural_donor_transition",
                    "noise_donor_transition",
                    "recipient_persistence",
                    "donor_map_persistence",
                )
            ]:
                raise ValueError("Inconsistent transplant score")


def verify(root):
    index = read(root / "checksums.json")
    for key, h in index.items():
        if hashlib.sha256((root / key).read_bytes()).hexdigest() != h:
            raise ValueError(f"Digest mismatch {key}")
    summary = read(root / "results.json")
    for name, run in summary["arms"].items():
        for item in run["evaluations"]:
            r = read(root / item["path"])
            counts, consistent, n = score(r)
            for cond, vals in counts.items():
                m = item["scores"]["conditions"][cond]
                for j, (acc, traj) in enumerate(zip(m["accuracy"], m["trajectory"]), 1):
                    if (
                        round(acc * n) != vals[f"hop{j}"]
                        or round(traj * n) != vals[f"path{j}"]
                    ):
                        raise ValueError("Score mismatch")
                    if cond != "canonical" and (
                        round(m["consistent"][j - 1] * n) != consistent[cond][f"hop{j}"]
                        or round(m["both_correct"][j - 1] * n)
                        != consistent[cond][f"both{j}"]
                    ):
                        raise ValueError("Unpaired consistency")
        if run.get("deep"):
            score(read(root / run["deep"]))
        if run.get("causal"):
            patches(read(root / run["causal"]))
        print(name, "verified")
    print(len(index), "SHA-256 digests verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root", type=Path, nargs="?", default=Path("docs/measurements/phase6")
    )
    verify(p.parse_args().root)
