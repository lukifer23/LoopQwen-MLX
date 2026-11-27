"""Inspect all immutable materialized first-visit arrays while the native trainer owns them."""

import argparse
import json
import struct
import time
from pathlib import Path

from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_reentry_fit import NAME, inputs
from scripts.assistance_deliberation import write_csv


def collect(folder):
    s = read(Path("research/experiments") / (NAME + ".json"))
    rows = inputs(tokenizer(s), s)
    start = time.monotonic()
    records = []
    for i, (k, (group, prefix, labels, _)) in enumerate(rows.items()):
        p = folder / f"{i}.safetensors"
        # A successive completed file certifies that mx.save_safetensors finished the prior write.
        signal = (
            folder / f"{i + 1}.safetensors"
            if i + 1 < len(rows)
            else Path("checkpoints/research") / NAME / "latest.json"
        )
        while not signal.exists():
            if (
                not folder.exists()
                or time.monotonic() - start > s["limits"]["wall_seconds"]
            ):
                raise RuntimeError("Native input cache closed before inspection")
            time.sleep(3)
        with p.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        item = header["hidden"]
        count = len(prefix) + len(labels) - 1
        assert (set(header) - {"__metadata__"}) == {"hidden"} and item["shape"] == [
            count,
            1024,
        ]
        assert item["dtype"] == "BF16" and item["data_offsets"] == [0, count * 1024 * 2]
        stat = p.stat()
        assert stat.st_size == count * 1024 * 2 + 8 + hlen
        records.append(
            {
                "view": k,
                "group": group,
                "index": i,
                "sequence_length": count,
                "size_bytes": stat.st_size,
                "completed_unix_ns": stat.st_mtime_ns,
                "sha256": digest(p),
            }
        )
    times = [r["completed_unix_ns"] for r in records]
    assert times == sorted(times)
    dest = Path("paper/reentry-feature-cache.csv")
    write_csv(dest, records)
    seconds = (times[-1] - times[0]) / 1e9
    size = sum(r["size_bytes"] for r in records)
    tokens = sum(r["sequence_length"] for r in records)
    summary = {
        "description": "Native first-visit BF16 arrays for complete training turns. The timestamp span runs from the first completed file to the last and includes writes and scheduling. All speed denominators exclude the first already-written file.",
        "registration": digest(Path("research/experiments") / (NAME + ".json")),
        "files": len(records),
        "positions": tokens,
        "width": 1024,
        "stored_bytes": size,
        "elapsed_first_to_last_s": seconds,
        "positions_per_s_after_first": (tokens - records[0]["sequence_length"])
        / seconds,
        "rows": digest(dest),
    }
    receipt = Path("docs/measurements/table-reentry-feature-cache.json")
    if receipt.is_file():
        assert read(receipt) == summary
    else:
        save(receipt, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native_cache", type=Path)
    collect(p.parse_args().native_cache)
