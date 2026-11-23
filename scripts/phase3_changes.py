"""Identify correctness flips by joining fresh 250K generation records on problem ID."""

import gzip
import json
from pathlib import Path


def _records(archive, arm, mode):
    outputs = {}
    for path in sorted((archive / "evaluations/250000" / arm / mode).glob("*.json.gz")):
        data = json.loads(gzip.decompress(path.read_bytes()))
        split = data["identity"]["decoding"]["split"]
        if split not in ("development", "development_ood"):
            continue
        if split in outputs:
            raise ValueError(f"Ambiguous evaluation identity for {mode}/{split}")
        records = data["generations"]["records"]
        by_id = {r["id"]: r for r in records}
        if len(by_id) != len(records):
            raise ValueError(f"Duplicate evaluation items in {path}")
        outputs[split] = by_id
    return outputs


def changes(archive, arm):
    bypass, gated = [_records(archive, arm, mode) for mode in ("bypass", "trained")]
    result = []
    for split in ("development", "development_ood"):
        if bypass[split].keys() != gated[split].keys():
            raise ValueError("Mismatched bypass/gated development cohort")
        for item, prior in bypass[split].items():
            rec = gated[split][item]
            if prior["correct"] == rec["correct"]:
                continue
            result.append(
                dict(
                    split=split,
                    id=item,
                    prompt=rec["prompt"],
                    family=rec["category"],
                    gold=rec["gold"],
                    effect="fixed" if rec["correct"] else "broke",
                    bypass=prior,
                    gated=rec,
                )
            )
    return result
