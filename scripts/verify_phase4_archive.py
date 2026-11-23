"""Verify the published Phase 4 files and recount the traced development scores on CPU."""

import gzip
import hashlib
import json
from pathlib import Path


def verify_files(root, index):
    for name, expected in index.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {root / name}")


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / "docs/measurements/phase4"
    verify_files(archive, json.loads((archive / "checksums.json").read_text()))
    adapters = json.loads((archive / "adapter-checksums.json").read_text())
    verify_files(root / "adapters/phase4", adapters)
    published_weights = {
        sha for name, sha in adapters.items() if name.endswith(".safetensors")
    }
    identities = set()
    cohorts = {}
    runs = json.loads((archive / "study-status.json").read_text())[
        "completed_evaluations"
    ]
    print("Checkpoint / depth                 Correct   Verified")
    for name in runs:
        with gzip.open(archive / "eval/primary" / (name + ".json.gz"), "rt") as src:
            result = json.load(src)
        identities.add(result["identity"]["weights"])
        correct = verified = count = 0
        for split in ("development/trace", "development_ood/trace"):
            score = result["scores"][split]
            rows = score["records"]
            ids = [(r["id"], r["prompt"], r["gold"]) for r in rows]
            if ids != cohorts.setdefault(split, ids) or len(
                {r["id"] for r in rows}
            ) != len(rows):
                raise ValueError(f"Evaluation cohort mismatch for {name} {split}")
            answers = sum(r["correct"] for r in rows)
            proofs = sum(r["verified"] for r in rows)
            if (answers, proofs, len(rows)) != (
                score["correct"],
                score["verified"],
                score["n"],
            ):
                raise ValueError(f"Score mismatch for {name} {split}")
            correct += answers
            verified += proofs
            count += len(rows)
        print(f"{name:32} {correct:3}/{count}   {verified:3}/{count}")
    if not published_weights.issubset(identities):
        raise ValueError("Published adapter has no matching evaluated weight digest")


if __name__ == "__main__":
    main()
