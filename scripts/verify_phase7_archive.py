"""Verify Phase 7 files and independently rescore paired trajectories and visible numbers."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.chat_turn7 import first_turn
from verify_phase6_archive import cohort, score, patches


def read(p):
    return json.loads(
        gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    )


def verify(root):
    inventory = read(root / "checksums.json")
    if set(inventory) != {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"
    }:
        raise ValueError("Archive inventory mismatch")
    for key, expected in inventory.items():
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
    t = read(root / "transfer/decision.json")
    reference = None
    for name, summary in t["verbal"].items():
        result = read(root / "transfer" / (name + ".json.gz"))
        current = [
            (r["id"], r["map"], r["start"], r["ids"], r["digit_targets"])
            for r in result["records"]
        ]
        if reference is not None and current != reference:
            raise ValueError("Transfer inputs differ")
        reference = current
        counts = [0] * len(summary["vocab"])
        bypass = 0
        for r in result["records"]:
            state = r["start"]
            for i in range(3):
                state = r["map"][state]
                if state != r["gold"][i]:
                    raise ValueError("Incorrect verbal target")
                if i < len(counts):
                    counts[i] += r["predicted_ids"][i] == r["digit_targets"][i]
            bypass += r["predicted_ids"][0] == r["digit_targets"][2]
        if (
            counts != summary["vocab"]
            or bypass != summary["first_exit_as_third"]
            or len(reference) != summary["n"]
        ):
            raise ValueError("Transfer scoring mismatch")
    if t["step"]:
        final_name = "transfer-" + str(t["step"])
        weights = read(root / "transfer" / (final_name + ".json.gz"))["weights"]
        if weights != inventory["transfer/model/adapters.safetensors"]:
            raise ValueError("Transfer weights mismatch")
        new = read(root / "transfer/orders.json.gz")
        if new["identity"]["weights"] != weights or cohort(new, s6) != cohort(
            child, s6
        ):
            raise ValueError("Incompatible post-transfer categorical cohort")
        score(new)
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
    turn = read(root / "language/first-turn.json")
    for name, result in turn["protocols"].items():
        rows = [
            first_turn(read(p), turn["chat_end"])
            for p in sorted((root / "language" / name).glob("*.json.gz"))
        ]
        if (
            rows != result["records"]
            or sum(r["correct"] for r in rows) != result["correct"]
            or sum(r["answer"] is not None for r in rows) != result["valid"]
        ):
            raise ValueError("First-turn rescore mismatch")
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
