"""Independently score archived categorical probabilities and patched predictions using map edges."""

import gzip
import hashlib
import json
import math
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/measurements/phase5"


def trajectory(row, n):
    mapping = row["map"]
    if sorted(mapping) != list(range(8)):
        raise ValueError("Invalid permutation")
    # Derive the entire orbit once; index cyclically to obtain labels at any hop.
    order = [row["start"]]
    while mapping[order[-1]] != order[0]:
        nxt = mapping[order[-1]]
        if nxt in order:
            raise ValueError("Invalid start orbit")
        order.append(nxt)
    return [order[i % len(order)] for i in range(1, n + 1)]


def rescore(result):
    depth = len(result["metrics"])
    correct = [0] * depth
    complete = [0] * depth
    consistent = [0] * depth
    for r in result["records"]:
        gold = trajectory(r, depth)
        pred = []
        for p in r["probabilities"]:
            if (
                len(p) != 8
                or any(not math.isfinite(q) or q < 0 for q in p)
                or abs(sum(p) - 1) > 1e-5
            ):
                raise ValueError("Invalid state probabilities")
            pred.append(max(range(8), key=p.__getitem__))
        if gold != r["gold"] or pred != r["pred"]:
            raise ValueError(
                "Stored trajectory disagrees with independent orbit or argmax"
            )
        for i in range(depth):
            correct[i] += pred[i] == gold[i]
            complete[i] += pred[: i + 1] == gold[: i + 1]
            consistent[i] += pred[i] == r["map"][pred[i - 1] if i else r["start"]]
    for i, m in enumerate(result["metrics"]):
        if (correct[i], complete[i], consistent[i], len(result["records"])) != (
            m["correct"],
            m["trajectory"],
            m["consistent"],
            m["n"],
        ):
            raise ValueError("Recorded state score disagrees")
    return correct


def verify(root=ARCHIVE):
    sums = json.loads((root / "checksums.json").read_text())
    for path, sha in sums.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Changed archive file " + path)
    for path in sorted((root / "predictions").glob("*.gz")):
        with gzip.open(path, "rt") as f:
            r = json.load(f)
        if "behavior" not in r:
            print(path.name, rescore(r))
            continue
        for kind, record in r["behavior"].items():
            rescore(record)
        groups = {}
        for item in r["intervention"]["records"]:
            current = trajectory(item["donor"], 1)[0]
            recipient = trajectory(item["recipient"], 2)[-1]
            target = item["recipient"]["map"][current]
            if target != item["expected"] or recipient != item["recipient_next"]:
                raise ValueError("Wrong patch target")
            for group in (
                ["all", "correct_donor"] if item["donor_read"] == current else ["all"]
            ):
                key = item["kind"] + "/" + group
                counts = groups.setdefault(key, [0] * 5)
                for i, x in enumerate(
                    (
                        1,
                        item["patch"] == target,
                        item["natural"] == target,
                        item["noise"] == target,
                        item["patch"] == recipient,
                    )
                ):
                    counts[i] += x
        for key in r["intervention"]["summary"]:
            x = r["intervention"]["summary"][key]
            if groups.get(key, [0] * 5) != [
                x[k]
                for k in (
                    "n",
                    "donor_transition",
                    "natural_donor_transition",
                    "noise_donor_transition",
                    "recipient_persistence",
                )
            ]:
                raise ValueError("Patched score mismatch")
        print(path.name, "counterfactuals and transplants verified")


if __name__ == "__main__":
    verify()
