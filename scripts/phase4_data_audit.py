"""Revalidate all new task labels and immutable split boundaries on the CPU."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.data.transitions import states
from loopqwen.evaluation.trace_verifier import verify
from loopqwen.io import digest, read, save


def audit():
    results = {}
    seen = set()
    closures = set()
    for split, meta in read("data/v4/manifest.json")["splits"].items():
        path = Path("data/v4") / (split + ".jsonl")
        if digest(path) != meta["sha256"]:
            raise ValueError("V4 tokenized bytes changed")
        verified = Counter()
        for line in path.open():
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError("Duplicate problem")
            seen.add(row["id"])
            if "closure" in row:
                if row["closure"] in closures:
                    raise ValueError("Repeated solved dependency graph")
                closures.add(row["closure"])
            if "transition" in row or "closure" in row:
                proof = verify(row, row["trace"])
                if not proof["verified"]:
                    raise ValueError(f"Invalid generated proof {row['id']}: {proof}")
                verified[row["category"]] += 1
        results[split] = dict(sha256=meta["sha256"], verified_new_examples=verified)
    maps = set()
    counts = {}
    baseline = [0, 0, 0]
    for split, sha in read("data/v4_pointer/manifest.json")["splits"].items():
        path = Path("data/v4_pointer") / (split + ".jsonl")
        if digest(path) != sha:
            raise ValueError("Pointer stream changed")
        count = 0
        for line in path.open():
            r = json.loads(line)
            key = tuple(r["map"])
            if key in maps or r["states"] != states(key, r["start"], 6):
                raise ValueError("Repeated map or inconsistent gold labels")
            for j, gold in enumerate(r["states"][:3]):
                conditioned = (r["start"], j)
                if split == "train":
                    counts.setdefault(conditioned, Counter())[gold] += 1
                else:
                    # Learn a deterministic start/depth prior using only training labels.
                    hist = counts[conditioned]
                    guess = max(range(8), key=lambda s: (hist[s], -s))
                    baseline[j] += int(guess == gold)
            maps.add(key)
            count += 1
        results["pointer/" + split] = dict(
            maps=count, explicit_state_labels=6 * count, sha256=sha
        )
    results["pointer/start_depth_prior"] = dict(
        terminal_correct=baseline,
        denominator=count,
        allowed_state_guess_probability=1 / 7,
    )
    registered = read("docs/measurements/phase3/v3-manifest.json")["splits"]
    for split in ("confirm", "confirm_ood"):
        expected = registered[split]["sha256"]
        if digest(Path("data/v3") / (split + ".jsonl")) != expected:
            raise ValueError("Historical sealed cohort bytes changed")
        results["historical/" + split] = dict(sha256=expected, check="file digest only")
    save("outputs/phase4/data-audit.json", results)
    return results


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
