"""Rescore archived state trajectories, paired orders and transplants using the standard library."""

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(p):
    data = p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix == ".gz" else data)


def identity(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cohort(result, spec):
    rows = []
    for r in result["records"]:
        if r["example_id"] != identity([r["map_id"], r["start"]]):
            raise ValueError("Incorrect example identity")
        order = list(range(len(r["map"])))
        condition = r["condition"]
        if condition not in spec["orders"]:
            raise ValueError("Unknown presentation")
        if condition.startswith("random"):
            random.Random(
                identity(
                    [
                        spec["evaluation_seed"],
                        0,
                        r["map_id"],
                        r["example_id"],
                        condition,
                    ]
                )
            ).shuffle(order)
        if condition == "reverse":
            order.reverse()
        if condition.startswith("rotate"):
            i = int(condition[6:]) % len(order)
            order = order[i:] + order[:i]
        if order != r["order"]:
            raise ValueError("Archived presentation differs from registration")
        rows.append([r["example_id"], condition, order, r["gold"]])
    return identity(rows)


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
    for condition, evaluation in result["behavior"].items():
        counts = [0] * 3
        for r in evaluation["records"]:
            node = r["start"]
            for j, probabilities in enumerate(r["probabilities"]):
                node = r["map"][node]
                prediction = max(
                    range(len(probabilities)), key=probabilities.__getitem__
                )
                if prediction != r["pred"][j] or node != r["gold"][j]:
                    raise ValueError(f"Inconsistent counterfactual record {condition}")
                counts[j] += prediction == node
        if counts != [m["correct"] for m in evaluation["metrics"]]:
            raise ValueError(f"Inconsistent counterfactual counts {condition}")
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


def reasoning_pair(folder, scores):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loopqwen.evaluation.answers import numeric

    records = defaultdict(dict)
    for file in folder.glob("*.json.gz"):
        r = read(file)
        g = r["generation"]
        ident = r["identity"]
        if (
            identity(r["prompt_ids"]) != ident["inputs"]
            or identity(r["item"]) != ident["question"]
        ):
            raise ValueError("Paired question or prompt digest changed")
        answer = None
        if "</think>" in g["text"]:
            answer = numeric(g["text"].rsplit("</think>", 1)[1].split("<|im_end|>")[0])
        if (
            answer != r["answer"]
            or (answer == numeric(r["item"]["gold"])) != r["correct"]
        ):
            raise ValueError("Incorrect visible-answer score")
        records[r["item"]["id"]][ident["arm"]] = r
    for pair in records.values():
        if (
            pair["base"]["prompt_ids"] != pair["recurrent"]["prompt_ids"]
            or pair["base"]["item"] != pair["recurrent"]["item"]
        ):
            raise ValueError("Different questions in a paired comparison")
    for key, summary in scores.items():
        cohort = [
            pair
            for pair in records.values()
            if key == "all" or pair["base"]["item"]["split"] == key
        ]
        if len(cohort) != summary["n"]:
            raise ValueError("Missing paired items")
        for arm in ("base", "recurrent"):
            xs = [pair[arm] for pair in cohort]
            result = [
                sum(x["correct"] for x in xs),
                sum(x["answer"] is not None for x in xs),
                sum(x["stop"] == "cap" for x in xs),
                sum(len(x["generation"]["token_ids"]) for x in xs),
            ]
            if result != [
                summary[arm][col] for col in ("correct", "valid", "capped", "tokens")
            ]:
                raise ValueError("Paired inference score mismatch")


def verify(root):
    index = read(root / "checksums.json")
    for key, h in index.items():
        if hashlib.sha256((root / key).read_bytes()).hexdigest() != h:
            raise ValueError(f"Digest mismatch {key}")
    summary = read(root / "results.json")
    spec = read(root / "registration.json")
    cohorts = {}
    for name, run in summary["arms"].items():
        for item in run["evaluations"]:
            r = read(root / item["path"])
            counts, consistent, n = score(r)
            final = item == run["evaluations"][-1]
            expected = spec["orders"] if final else ["canonical", "random0"]
            key = "final" if final else item["step"]
            cohort_id = cohort(r, spec)
            if (
                cohorts.setdefault(key, cohort_id) != cohort_id
                or list(counts) != expected
            ):
                raise ValueError("Unpaired arms or missing presentation conditions")
            if n != (spec["evaluation_maps"] if final else spec["early_items"]):
                raise ValueError("Incomplete evaluation cohort")
            randoms = [counts[k] for k in counts if k.startswith("random")]
            recorded = item["scores"]["conditions"]["mean_random"]
            for j in range(1, 4):
                for metric, prefix in (("accuracy", "hop"), ("trajectory", "path")):
                    expected_score = sum(c[f"{prefix}{j}"] for c in randoms) / (
                        n * len(randoms)
                    )
                    if abs(recorded[metric][j - 1] - expected_score) > 1e-10:
                        raise ValueError("Incorrect clustered random score")
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
    if "reasoning_pair" in summary:
        reasoning_pair(root / "reasoning-pair", summary["reasoning_pair"])
        print("Paired token prompts and visible-answer scores verified")
    print(len(index), "SHA-256 digests verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root", type=Path, nargs="?", default=Path("docs/measurements/phase6")
    )
    verify(p.parse_args().root)
