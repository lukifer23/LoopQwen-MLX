"""Verify Phase 7 files and independently rescore paired trajectories and visible numbers."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopqwen.evaluation.answers import numeric
from verify_phase6_archive import cohort, score, patches


def read(p):
    return json.loads(
        gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    )


def verify(root):
    for key, expected in read(root / "checksums.json").items():
        if hashlib.sha256((root / key).read_bytes()).hexdigest() != expected:
            raise ValueError(key)
    parent_dir = Path(__file__).resolve().parents[1] / "docs/measurements/phase6"
    s6 = read(parent_dir / "registration.json")
    d = read(root / "decision.json")
    st = read(root / "model/state.json")
    parent = read(parent_dir / "randomized/orders-480.json.gz")
    child = read(root / "endpoint.json.gz")
    if (
        cohort(parent, s6) != cohort(child, s6)
        or child["identity"]["weights"]
        != hashlib.sha256(
            (root / "model/adapters.safetensors").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Unmatched endpoint or weights")
    if (
        st["optimizer_step"] != st["step"]
        or st["step"] != 480 + len(st["history"])
        or st["lineage"]["optimizer"]
        != read(parent_dir / "results.json")["arms"]["randomized"]["optimizer"]
    ):
        raise ValueError("Invalid lineage")
    counts, _, n = score(child)
    old = score(parent)[0]
    randoms = [k for k in counts if k.startswith("random")]
    for j in range(1, 4):
        actual = sum(counts[k][f"hop{j}"] for k in randoms) / (len(randoms) * n)
        baseline = sum(old[k][f"hop{j}"] for k in randoms) / (len(randoms) * n)
        if abs(actual - baseline - d["paired"]["difference"][j - 1]) > 1e-10:
            raise ValueError("Paired score mismatch")
    score(read(root / "deep.json.gz"))
    if d["installed"]:
        patches(read(root / f"causal-{d['step']}.json.gz"))
    langs = read(root / "language/decision.json")
    for path in (root / "language").rglob("*.json.gz"):
        r = read(path)
        ids = r["token_ids"]
        if r["close_positions"] != [
            i for i, t in enumerate(ids) if t == r["closing_id"]
        ]:
            raise ValueError("Close token mismatch")
        valid = (
            len(r["close_positions"]) == 1
            and not r["open_positions"]
            and r["stop"] == "eos"
        )
        answer = numeric(r["visible"]) if valid else None
        if r["answer"] != answer or (r["status"] == "correct") != (
            answer == numeric(r["item"]["gold"])
        ):
            raise ValueError("Answer mismatch")
    for name, summary in langs["calibration"].items():
        records = [read(p) for p in (root / "language" / name).glob("*.json.gz")]
        if (
            len(records),
            sum(r["answer"] is not None for r in records),
            sum(r["status"] == "correct" for r in records),
        ) != (8, summary["valid"], summary["correct"]):
            raise ValueError("Calibration count mismatch")
    for base in (root / "language").glob("*/base/*.json.gz"):
        a = read(base)
        b = read(base.parent.parent / "candidate" / base.name)
        if a["prompt_ids"] != b["prompt_ids"] or a["item"] != b["item"]:
            raise ValueError("Unpaired comparison")
    print(
        f"R{st['step']} verified on {n} map clusters, {len(child['records'])} presentations; language scores and file digests verified."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "archive", nargs="?", type=Path, default=Path("docs/measurements/phase7")
    )
    verify(p.parse_args().archive)
