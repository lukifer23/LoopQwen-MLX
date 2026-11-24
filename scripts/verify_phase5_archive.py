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
    seen = set()
    for r in result["records"]:
        if (
            r["example_id"] in seen
            or r["map_id"] != hashlib.sha256(bytes(r["map"])).hexdigest()
        ):
            raise ValueError("Duplicate request or incorrect map identity")
        seen.add(r["example_id"])
        if len(r["probabilities"]) != depth:
            raise ValueError("Missing categorical prediction")
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
    if set(sums) != {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }:
        raise ValueError("Archive inventory differs")
    registration = json.loads((root / "registration.json").read_bytes())
    training = json.loads((root / "training.json").read_bytes())
    if not set(registration["arms"]).issubset(training):
        raise ValueError("Missing registered arm")
    cases = json.loads(gzip.decompress((root / "cases.json.gz").read_bytes()))
    cohorts = {}
    for path, sha in sums.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Changed archive file " + path)
    for path in sorted((root / "transfer").glob("*.gz")):
        rows = json.loads(gzip.decompress(path.read_bytes()))["records"]
        for r in rows:
            if trajectory(r, r["requested"])[-1] != r["gold"] or r["correct"] != (
                r["greedy_token_id"] == r["reference_token_id"]
            ):
                raise ValueError("Incorrect transfer gold or prediction")
    for path in sorted((root / "predictions").glob("*.gz")):
        with gzip.open(path, "rt") as f:
            r = json.load(f)
        if "behavior" not in r:
            if "step" in r:
                ids = [row["example_id"] for row in r["records"]]
                if cohorts.setdefault(r["step"], ids) != ids:
                    raise ValueError("Unpaired evaluation cohort")
            print(path.name, rescore(r))
            continue
        for kind, record in r["behavior"].items():
            rescore(record)
            for j, m in enumerate(record["metrics"]):
                agree = 0
                for c, base, altered in zip(
                    cases,
                    r["behavior"]["original"]["records"],
                    record["records"],
                    strict=True,
                ):
                    if (altered["map"], altered["start"], altered["order"]) != (
                        c[kind]["map"],
                        c[kind]["start"],
                        c[kind]["order"],
                    ):
                        raise ValueError("Counterfactual request mismatch")
                    pred = base["pred"][j]
                    agree += (
                        c["perm"][pred] if kind == "relabel" else pred
                    ) == altered["pred"][j]
                if agree != m["prediction_agreement"]:
                    raise ValueError("Incorrect counterfactual consistency")
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
