"""Verify actual full prefix files for every native visit without loading a model."""

import argparse
import fcntl
import hashlib
from pathlib import Path

from safetensors import safe_open

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_documents import FROZEN


def uncached_digest(path):
    with path.open("rb") as f:
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(f, fcntl.F_NOCACHE, 1)
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit(book, stage, loop, ordinary):
    cfg = read(FROZEN)[book]
    result = {}
    files = {}
    for arm, policy in (("loop", loop), ("ordinary", ordinary)):
        reg = read(policy / "registration.json")
        assert digest(policy / "adapters.safetensors") == reg["weights"]
        visits = reg["serving_depth"]
        sites = (
            [(i, 0) for i in range(12)]
            + [(i, n) for n in range(visits) for i in range(12, 16)]
            + [(i, 0) for i in range(16, 24)]
        )
        key = fingerprint([reg["weights"], sites, cfg["common_sha"], "saved-prefix-v1"])
        directory = Path("outputs/middle-depth/documents") / key
        receipt = read(ROOT / "prefix-retention" / ("documents-" + key + ".json"))
        assert receipt["complete"] and not receipt["evicted"]
        records = [
            v for g in read(directory / "prefix.json")["groups"] for v in g["states"]
        ]
        assert [tuple(r["site"]) for r in records] == sites
        rows = []
        branch = {}
        for site, row in zip(sites, records, strict=True):
            file = directory / row["file"]
            value = uncached_digest(file)
            assert value == row["sha256"] == receipt["states"][row["file"]]
            with safe_open(str(file), framework="np") as f:
                tensor_names = list(f.keys())
                shapes = {k: f.get_slice(k).get_shape() for k in tensor_names}
                kinds = {k: f.get_slice(k).get_dtype() for k in tensor_names}
            assert row["consumed"] == cfg["common"]
            if site[0] % 4 == 3:
                assert set(shapes) == {"k", "v"} and all(
                    v[2] == cfg["common"] for v in shapes.values()
                )
            else:
                assert set(shapes) == {"convolution", "delta"}
            if site[1] > 0:
                first = branch[site[0], 0]
                assert (
                    not first.samefile(file) and value != receipt["states"][first.name]
                )
            branch[site] = file
            rows.append(
                {
                    "site": site,
                    "kind": row["kind"],
                    "consumed": row["consumed"],
                    "shapes": shapes,
                    "dtypes": kinds,
                    "bytes": file.stat().st_size,
                    "sha256": value,
                }
            )
        files[arm] = branch
        result[arm] = {"weights": reg["weights"], "prefix": key, "visits": rows}
    identical = [
        {"site": site, "bytes": p.stat().st_size}
        for site, p in files["ordinary"].items()
        if p.samefile(files["loop"][site])
    ]
    assert {tuple(r["site"]) for r in identical}.issuperset({(n, 0) for n in range(12)})
    save(
        ROOT / "prefix-audits" / (stage + "-" + book + ".json"),
        {
            "models": result,
            "common_input": cfg["common_sha"],
            "actual_prefix_tokens": cfg["common"],
            "shared_immutable_files": identical,
            "avoided_bytes": sum(r["bytes"] for r in identical),
            "passed": True,
        },
    )
    print(
        book,
        cfg["common"],
        "tokens; physical sites",
        *(len(result[a]["visits"]) for a in ("ordinary", "loop")),
        "shared bytes",
        sum(r["bytes"] for r in identical),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("book", choices=("anne", "book"))
    p.add_argument("--loop", type=Path, required=True)
    p.add_argument("--ordinary", type=Path, required=True)
    a = p.parse_args()
    audit(a.book, a.stage, a.loop, a.ordinary)
