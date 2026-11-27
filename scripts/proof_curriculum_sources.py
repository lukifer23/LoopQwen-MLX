"""Preserve the original sampled logic records for a compact independent offline replay."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from loopqwen.io import digest, read, save

SOURCE = Path("research/sources/proof-curriculum")
INVENTORY = Path("docs/measurements/proof-curriculum-curated-sources.json")


def store(path, raw):
    path.parent.mkdir(exist_ok=True, parents=True)
    compressed = gzip.compress(raw, mtime=0)
    if path.exists():
        assert gzip.decompress(path.read_bytes()) == raw
    else:
        path.write_bytes(compressed)


def build():
    original = Path("data/research_external/proofwriter-V2020.12.3.zip")
    spec = read("research/experiments/proof-curriculum-01.json")
    prev = read("research/experiments/reasoned-evidence-01.json")
    data = read(spec["training"])
    by_world = {w["theory"]: w for w in data["proof"]}
    assert digest(original) == spec["sources"]["sha256"]
    with ZipFile(original) as z, z.open(data["member"]) as stream:
        lines = [line for line in stream if json.loads(line)["id"] in by_world]
    assert len(lines) == len(by_world)
    oracle = {
        r["id"]: r["world_sha256"]
        for r in read("docs/measurements/proof-curriculum-source-oracle.json")[
            "results"
        ]
    }
    for line in lines:
        world = json.loads(line)
        assert hashlib.sha256(line).hexdigest() == oracle[by_world[world["id"]]["id"]]
    logic = SOURCE / "proofwriter-train-worlds.jsonl.gz"
    store(logic, b"".join(lines))
    selected = {w["id"]: w for w in data["human"]}
    human = prev["sources"]["esnli"]
    assert digest(Path(human["path"])) == human["sha256"]
    source_rows = []
    with Path(human["path"]).open(newline="") as f:
        for r in csv.DictReader(f):
            w = selected.get(r["pairID"])
            if w is None:
                continue
            assert tuple(w[k] for k in ("evidence", "claim", "reason")) == tuple(
                r[k].strip() for k in ("Sentence1", "Sentence2", "Explanation_1")
            )
            assert (
                w["label"]
                == {
                    "neutral": "uncertain",
                    "entailment": "true",
                    "contradiction": "false",
                }[r["gold_label"]]
            )
            source_rows.append(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    assert len(source_rows) == 384
    hfile = SOURCE / "esnli-train-annotations.jsonl.gz"
    store(hfile, "".join(source_rows).encode())
    dev = prev["sources"]["folio_validation"]
    p = Path(dev["path"])
    assert digest(p) == dev["sha256"]
    folio = SOURCE / "folio-validation.jsonl.gz"
    store(folio, p.read_bytes())
    result = {
        "proofwriter": {
            "full_zip_sha256": digest(original),
            "member": data["member"],
            "count": len(lines),
            "selected_original_lines": digest(logic),
        },
        "human": {
            "full_csv_sha256": human["sha256"],
            "count": len(source_rows),
            "selected_original_fields": digest(hfile),
        },
        "folio": {
            "original_sha256": dev["sha256"],
            "count": len(p.read_bytes().splitlines()),
            "compressed": digest(folio),
        },
    }
    if INVENTORY.exists():
        assert read(INVENTORY) == result
    else:
        save(INVENTORY, result)
    print(
        "Original logic records, complete human annotation rows, complete FOLIO development file:",
        len(lines),
        len(source_rows),
        result["folio"]["count"],
    )


if __name__ == "__main__":
    build()
