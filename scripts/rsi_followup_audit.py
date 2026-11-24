"""Verify terminal-only versus verified traces and independently rescore the time control."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from rsi_archive import sha, read

ROOT = Path("docs/measurements")


def traces():
    root = ROOT / "rsi_trace1"
    items = read(root / "candidates.json.gz")
    task = read(root / "splits.json")
    seen = {r["map_id"] for r in task["train"]}
    if seen.intersection(r["map_id"] for r in task["development"]):
        raise ValueError("Trace split overlaps")
    valid = set()
    for r in items:
        if (
            r["map_id"] not in seen
            or hashlib.sha256(bytes(r["map"])).hexdigest() != r["map_id"]
        ):
            raise ValueError("Trace map")
        n = r["start"]
        trajectory = []
        for _ in range(3):
            n = r["map"][n]
            trajectory.append(n)
        parts = re.split(r"\s*(?:,|->|→)\s*", r["response"].strip().rstrip(".!"))
        verified = (
            len(parts) == 3
            and all(v in "01234567" and len(v) == 1 for v in parts)
            and list(map(int, parts)) == trajectory
            and r["stop"] != "cap"
        )
        if verified:
            valid.add(r["map_id"])
        if verified != r["verified"]:
            raise ValueError("Trace verdict")
    d = read(root / "decision.json")
    if len(items) != d["candidates"] or len(valid) != d["verified_maps"]:
        raise ValueError("Trace yield")
    print(len(valid), "maps with verified generated traces of", len(seen))


def time_score(root):
    d = read(root / "decision.json")
    ev = read(root / "evaluation.json.gz")
    ref = read(ROOT / "rsi1/parent-evaluation.json.gz")["records"]
    counts = [0] * 6
    complete = 0
    for r, old in zip(ev["records"], ref, strict=True):
        if (r["map"], r["start"], r["order"]) != (
            old["map"],
            old["start"],
            old["order"],
        ):
            raise ValueError("Unpaired time control")
        state = r["start"]
        hits = []
        for i, pr in enumerate(r["probabilities"]):
            state = r["map"][state]
            hits.append(max(range(8), key=pr.__getitem__) == state)
            counts[i] += hits[-1]
        complete += all(hits)
    if [k / len(ref) for k in counts] != d["evaluation"]["accuracy"] or complete / len(
        ref
    ) != d["evaluation"]["path"][-1]:
        raise ValueError("Time-control scores")
    if (
        sha(root / "model/adapters.safetensors") != ev["weights"]
        or ev["weights"] != d["weights"]
    ):
        raise ValueError("Time-control weights")
    print("Time-matched sixth hops:", counts[-1], "/", len(ref))


def archive_time():
    dest = ROOT / "rsi_time1"
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=ROOT) as tmp:
        p = Path(tmp) / "release"
        shutil.copytree("outputs/rsi_time1", p)
        d = read(p / "decision.json")
        g = Path(d["generation"])
        (p / "model").mkdir()
        for name in ("adapters.safetensors", "state.json"):
            shutil.copy(g / name, p / "model" / name)
        shutil.copy("experiments/rsi_followup.json", p / "registration.json")
        time_score(p)
        p.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive-time", action="store_true")
    a = p.parse_args()
    traces()
    if a.archive_time:
        archive_time()
    else:
        time_score(ROOT / "rsi_time1")
