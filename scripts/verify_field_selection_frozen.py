"""Reconstruct both registered trajectories and scores from a restored, offline research checkout."""

import argparse
from pathlib import Path

from loopqwen.io import digest, read
from loopqwen.research.field_selection import verify
from loopqwen.research.registered import registered


def run(root):
    spec = registered("field-selection-01")
    assert read(root / "registration.json") == spec
    expected = read(root / "checksums.json")
    registry = read("research/registry.json")["experiments"][spec["id"]]
    assert digest(root / "checksums.json") == registry["archive"]
    actual = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    assert actual == expected
    decision = read(root / "decision.json")
    verify(root, decision)
    print(
        "Reconstructed every registered request on both models, real tool returns, recorded scores and the exact paired decision."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("research/results/field-selection-01")
    )
    run(p.parse_args().root)
