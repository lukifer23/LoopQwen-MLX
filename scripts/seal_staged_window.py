"""Seal a completed native arm before releasing its bounded window; preserve masked review."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read
from loopqwen.research.registered import registered
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

STAGE = "staged-query-transfer-01"


def check(name, window, *, seal=False):
    spec = registered(name)
    dest = Path(spec["intermediate_window_archive"])
    root = Path("outputs/research") / name
    assert name == STAGE and not (root / "decision.json").exists()
    assert read(root / "candidate-window.json")["window"] == window
    source = root / "candidate.json.gz"
    artifact = dest / "candidate.json.gz"
    if seal:
        dest.mkdir(exist_ok=True, parents=True)
        if not artifact.exists():
            shutil.copyfile(source, artifact)
    else:
        committed(artifact)
    assert digest(artifact) == digest(source)
    rows = unpack(Path(spec["material"]))["rows"]
    records = unpack(artifact)
    index = read(root / "request-index.json")
    assert not index["control"] and len(records) == len(rows) == len(index["candidate"])
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / "item.json.gz"
        for w, r, sha in zip(rows, records, index["candidate"], strict=True):
            assert w["id"] == r["id"] and len(r["requests"]) == 1
            q = r["requests"][0]
            turns = q["trajectory"]["turns"]
            assert turns[0]["prompt_ids"] == w["initial_ids"]
            assert (
                q["read_only_visits"]
                == 2 * q["output_positions"]
                == 2 * sum(len(t["token_ids"]) for t in turns)
            )
            for t in turns:
                n = len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                assert len(t["kv_history"]) == 6
                assert all(s["offset"] == s["retained"] == n for s in t["kv_history"])
                assert bool(t.get("scratch_bytes")) == (
                    len(w["initial_ids"]) >= spec["stage_from_tokens"]
                )
            packed(f, r)
            assert digest(f) == sha
    data = {
        "job": name,
        "window": window,
        "arm": "candidate",
        "requests": len(records),
        "records": digest(artifact),
        "model": read(root / "weights.json"),
        "registration": digest(root / "registration.json"),
        "implementation": digest(root / "environment.json"),
        "material": digest(Path(spec["material"])),
        "index": digest(root / "request-index.json"),
    }
    receipt = dest / "receipt.json"
    if seal:
        persist(receipt, data)
    else:
        committed(receipt)
        assert read(receipt) == data
    return {"job": name, "archive": str(receipt), "sha256": digest(receipt)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    print(
        check(
            STAGE,
            int(read("outputs/research/active-session.json")["start"]),
            seal=not a.verify,
        )
    )
