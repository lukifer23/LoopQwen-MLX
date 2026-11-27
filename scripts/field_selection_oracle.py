"""Cross-check actual release wheels and frozen CSV cells without the model or tool provider."""

import argparse
import csv
import hashlib
import io
import sqlite3
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from loopqwen.io import digest, read, save

PANEL = Path("research/data/field-selection-panel.json")
TRAIN = PANEL.with_name("field-selection-training.json")
CACHE = Path("data/research_external/field_selection/wheels")
OUTPUT = Path("docs/measurements/field-selection-oracle.json")


def wheel(w, download):
    path = CACHE / w["wheel"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Use --acquire to obtain {path.name}")
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(w["wheel_url"], timeout=60) as stream:
            temp = path.with_suffix(".part")
            with temp.open("wb") as target:
                while data := stream.read(1 << 20):
                    target.write(data)
            assert digest(temp) == w["wheel_sha256"]
            temp.replace(path)
    assert digest(path) == w["wheel_sha256"]
    with ZipFile(path) as z:
        (member,) = [m for m in z.namelist() if m.endswith(".dist-info/METADATA")]
        raw = z.read(member)
    assert (
        raw == Path(w["path"]).read_bytes()
        and hashlib.sha256(raw).hexdigest() == w["metadata_sha256"]
    )
    head = BytesParser().parsebytes(raw)
    assert (head["Name"], head["Version"], head["Requires-Python"]) == (
        w["package"],
        w["version"],
        w["requirement"],
    )
    return {
        "package": w["package"],
        "wheel": path.name,
        "artifact_bytes": path.stat().st_size,
        "sha256": digest(path),
        "metadata_member": member,
        "metadata_sha256": w["metadata_sha256"],
    }


def select(raw, search_col, key, result_col):
    header, *records = raw
    assert len(set(header)) == len(header)
    # Bind all untrusted data as parameters. Use indices for SQL identifiers.
    search, answer = header.index(search_col), header.index(result_col)
    db = sqlite3.connect(":memory:")
    try:
        cols = ",".join(f"c{i} TEXT" for i in range(len(header)))
        db.execute(f"CREATE TABLE records ({cols})")
        db.executemany(
            "INSERT INTO records VALUES (" + ",".join("?" * len(header)) + ")", records
        )
        return [
            v
            for (v,) in db.execute(
                f"SELECT c{answer} FROM records WHERE c{search} = ?", (key,)
            )
        ]
    finally:
        db.close()


def run(download=False):
    panel = read(PANEL)
    with ThreadPoolExecutor(max_workers=3) as executor:
        wheels = list(executor.map(lambda w: wheel(w, download), panel["wheels"]))
    lookups = Counter()
    for w in panel["rows"]:
        if w["family"] not in ("file", "wheel"):
            continue
        key = w["calls"][0]["arguments"]
        table = list(csv.reader(io.StringIO(panel["documents"][w["document"]])))
        assert select(table, key["key_column"], key["key"], key["value_column"]) == [
            w["observation"]["value"]
        ]
        lookups[w["family"]] += 1
    reversed_count = 0
    for w in read(TRAIN)["roles"]:
        source = [w["header"], *w["rows"]]
        for a, b, x, y in [
            (w["a"], w["b"], w["x"], w["y"]),
            (w["b"], w["a"], w["y"], w["x"]),
        ]:
            assert select(source, a, x, b) == [y] and select(source, b, x, a) == []
            reversed_count += 1
    report = {
        "panel": digest(PANEL),
        "training": digest(TRAIN),
        "held_real_wheels": wheels,
        "sqlite_held": dict(lookups),
        "opposite_training_roles": reversed_count,
        "sqlite_actual_missing_swaps": reversed_count,
    }
    if OUTPUT.exists():
        assert read(OUTPUT) == report
    else:
        save(OUTPUT, report)
    print(
        len(wheels),
        "original wheels,",
        sum(lookups.values()),
        "held cells,",
        reversed_count,
        "opposite-direction training lookups and actual reversal failures",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    run(parser.parse_args().acquire)
