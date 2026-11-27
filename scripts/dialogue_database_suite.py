"""Qualify original human SQL on every published alternate database for the held schemas."""

import argparse
import hashlib
import shutil
import sqlite3
import urllib.request
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

from loopqwen.io import digest, read, save
from scripts.dialogue_database_sources import CACHE, DEST

URL = "https://drive.usercontent.google.com/download?id=1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w&export=download&confirm=t"
SUITE = CACHE.with_name("distilled.zip")
SHA = "9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96"
FILES = CACHE.parent / "distilled"
MANIFEST = DEST / "distilled.json"


def qualify(fetch=False):
    if fetch and not SUITE.is_file():
        pending = SUITE.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=90) as r, pending.open("wb") as f:
            shutil.copyfileobj(r, f)
        if digest(pending) != SHA:
            raise ValueError(
                "Distilled suite does not match the published bytes used here"
            )
        pending.replace(SUITE)
    assert digest(SUITE) == SHA
    output = {}
    with ZipFile(SUITE) as z:
        for db in read(DEST / "originals.json")["cohorts"]["dev"]:
            name = db["database"]
            turns = db["dialogues"][0]["sequence"]
            members = sorted(
                p
                for p in z.namelist()
                if p.startswith(f"database/{name}/") and p.endswith(".sqlite")
            )
            assert any(Path(p).stem == name for p in members)
            cases = []
            for member in members:
                path = FILES / name / Path(member).name
                path.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                # Extract and hash one source at a time, including the large tennis database.
                with z.open(member) as src:
                    if path.is_file():
                        with path.open("rb") as existing:
                            while chunk := src.read(1 << 20):
                                assert existing.read(len(chunk)) == chunk
                                checksum.update(chunk)
                            assert not existing.read(1)
                    else:
                        pending = path.with_suffix(".partial")
                        with pending.open("wb") as out:
                            while chunk := src.read(1 << 20):
                                checksum.update(chunk)
                                out.write(chunk)
                        pending.replace(path)
                with closing(
                    sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                ) as con:
                    result = [list(map(list, con.execute(q["query"]))) for q in turns]
                cases.append(
                    {
                        "member": member,
                        "sha256": checksum.hexdigest(),
                        "results": result,
                    }
                )
            differing = [
                sum(c["results"][j] != [[q["returned"]]] for c in cases)
                for j, q in enumerate(turns)
            ]
            output[name] = {"databases": cases, "different_from_original": differing}
    n = sum(len(g["databases"]) for g in output.values())
    report = {
        "source": URL,
        "sha256": SHA,
        "originals_sha256": digest(DEST / "originals.json"),
        "schemas": len(output),
        "database_instances": n,
        "executed_queries": 2 * n,
        "questions_with_changed_results": sum(
            v > 0 for g in output.values() for v in g["different_from_original"]
        ),
        "by_schema": output,
    }
    if MANIFEST.exists():
        assert read(MANIFEST) == report
    else:
        save(MANIFEST, report)
    print({k: v for k, v in report.items() if k != "by_schema"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    qualify(p.parse_args().acquire)
