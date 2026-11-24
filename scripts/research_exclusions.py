"""Freeze verified historical map exclusions, including unpublished sealed ID inventories."""

from pathlib import Path
import gzip
from loopqwen.data import curriculum as c
from loopqwen.io import digest, read, save
from loopqwen.identity import fingerprint


def main():
    root = Path("research/data")
    used = c.historical()
    groups = {}
    for family, key in [
        ("phase5_pointer", "files"),
        ("v4_pointer", "splits"),
        ("v4", "splits"),
    ]:
        directory = Path("data") / family
        for k, expected in read(directory / "manifest.json")[key].items():
            p = directory / (k + ".jsonl")
            sha = digest(p)
            if sha != (expected["sha256"] if isinstance(expected, dict) else expected):
                raise ValueError(str(p))
            groups[str(p)] = sha
    for name, reg in [("rsi1", "rsi_pilot"), ("rsi2", "rsi_replication")]:
        arc = Path("docs/measurements") / name
        if name == "rsi2":
            trace = read("docs/measurements/rsi_trace1/splits.json")
            used.update(r["map_id"] for rows in trace.values() for r in rows)
        splits, _ = c._pools(read(f"experiments/{reg}.json"), used.copy())
        ids = {k: fingerprint(rows) for k, rows in splits.items()}
        if ids != read(arc / "data.json")["identities"]:
            raise ValueError("Incomplete historical map inventory")
        used.update(r["map_id"] for rows in splits.values() for r in rows)
    payload = gzip.compress(("\n".join(sorted(used)) + "\n").encode(), mtime=0)
    p = root / "historical-maps.txt.gz"
    if p.exists() and p.read_bytes() != payload:
        raise ValueError("Inventory changed")
    p.write_bytes(payload)
    save(
        root / "historical.json",
        dict(count=len(used), sha256=digest(p), sources=groups),
    )


if __name__ == "__main__":
    main()
