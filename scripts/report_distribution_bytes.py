"""Independently verify each enrolled release file against downloaded archive bytes and metadata."""

import argparse
import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

from packaging.utils import canonicalize_name

from loopqwen.io import digest, read, save

PANEL = Path("research/data/report-action-panel.json")
RECEIPT = Path("docs/measurements/report-distribution-bytes.json")


def targets(panel):
    for row in panel["rows"]:
        if row["family"] == "inventory":
            small, large = row["keys"]
            records = {
                r["filename"]: r
                for r in csv.DictReader(
                    io.StringIO(panel["documents"][row["document"]])
                )
            }
            yield (
                row["cluster"],
                {small, large},
                row["questions"][0],
                *row["gold"],
                records,
            )
        elif row["family"] == "distribution" and not row["expression"]:
            (other,) = [
                r
                for r in panel["rows"]
                if r["cluster"] == row["cluster"] and r["expression"]
            ]
            yield (
                row["cluster"],
                other["questions"][0].rsplit("\n\n", 1)[1],
                row["questions"][0].rsplit("\n\n", 1)[1],
                row["gold"],
                other["gold"],
                {},
            )


def verify(fetch=False, *, panel_path=PANEL, receipt=RECEIPT):
    panel = read(panel_path)
    lock = subprocess.check_output(["git", "show", panel["lock_revision"] + ":uv.lock"])
    assert hashlib.sha256(lock).hexdigest() == panel["lock_sha256"]
    packages = {p["name"]: p for p in tomllib.loads(lock.decode())["package"]}
    prior = read(receipt) if receipt.is_file() else None
    if not fetch and prior is None:
        raise FileNotFoundError(receipt)
    saved = {} if prior is None else {r["filename"]: r for r in prior["files"]}
    evidence = []
    for group, change, lookup, first_answer, difference, records in targets(panel):
        pkg = packages[group]
        selected = []
        for w in pkg["wheels"]:
            name = unquote(urlsplit(w["url"]).path.rsplit("/", 1)[1])
            if name not in change:
                continue
            if fetch:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / name
                    with (
                        urlopen(w["url"], timeout=60) as response,
                        path.open("wb") as stream,
                    ):
                        shutil.copyfileobj(response, stream)
                    with ZipFile(path) as z:
                        (meta,) = [
                            p for p in z.namelist() if p.endswith(".dist-info/METADATA")
                        ]
                        metadata = BytesParser().parsebytes(
                            z.read(meta), headersonly=True
                        )
                    r = {
                        "package": metadata["Name"],
                        "version": metadata["Version"],
                        "filename": name,
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "source": w["url"],
                    }
            else:
                r = saved[name]
            assert r["filename"] == name and r["source"] == w["url"]
            assert (
                canonicalize_name(r["package"]) == group
                and r["version"] == pkg["version"]
            )
            assert r["bytes"] == w["size"] and "sha256:" + r["sha256"] == w["hash"]
            if records:
                assert records[name]["sha256"] == r["sha256"] and records[name][
                    "bytes"
                ] == str(r["bytes"])
            selected.append(r)
        assert len(selected) == 2
        small, large = sorted(selected, key=lambda r: r["bytes"])
        assert small["filename"] in lookup and large["filename"] not in lookup
        assert str(small["bytes"]) == first_answer
        assert str(large["bytes"] - small["bytes"]) == difference
        evidence.extend([small, large])
    assert len(evidence) == 16
    report = {
        "panel_sha256": digest(panel_path),
        "lock_sha256": panel["lock_sha256"],
        "total_bytes": sum(r["bytes"] for r in evidence),
        "files": evidence,
    }
    if prior:
        assert prior == report
    else:
        save(receipt, report)
    print(len(evidence), "original wheels,", report["total_bytes"], "physical bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download and independently measure the locked originals.",
    )
    parser.add_argument("--keyed-inventories", action="store_true")
    args = parser.parse_args()
    if args.keyed_inventories:
        verify(
            args.fetch,
            panel_path=Path("research/data/instruction-interpolation-panel.json"),
            receipt=Path("docs/measurements/keyed-inventory-bytes.json"),
        )
    else:
        verify(args.fetch)
