"""Independently verify each enrolled release file against downloaded archive bytes and metadata."""

import argparse
import hashlib
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


def verify(fetch=False):
    panel = read(PANEL)
    lock = subprocess.check_output(["git", "show", panel["lock_revision"] + ":uv.lock"])
    assert hashlib.sha256(lock).hexdigest() == panel["lock_sha256"]
    packages = {p["name"]: p for p in tomllib.loads(lock.decode())["package"]}
    prior = read(RECEIPT) if RECEIPT.is_file() else None
    if not fetch and prior is None:
        raise FileNotFoundError(RECEIPT)
    saved = {} if prior is None else {r["filename"]: r for r in prior["files"]}
    evidence = []
    for group in dict.fromkeys(
        w["cluster"] for w in panel["rows"] if w["family"] == "distribution"
    ):
        pkg = packages[group]
        direct, derived = [
            w
            for w in panel["rows"]
            if w["family"] == "distribution" and w["cluster"] == group
        ]
        lookup, change = [
            w["questions"][0].rsplit("\n\n", 1)[1] for w in (direct, derived)
        ]
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
            selected.append(r)
        assert len(selected) == 2
        small, large = sorted(selected, key=lambda r: r["bytes"])
        assert small["filename"] in lookup and large["filename"] not in lookup
        assert str(small["bytes"]) == direct["gold"]
        assert str(large["bytes"] - small["bytes"]) == derived["gold"]
        evidence.extend([small, large])
    assert len(evidence) == 16
    report = {
        "panel_sha256": digest(PANEL),
        "lock_sha256": panel["lock_sha256"],
        "total_bytes": sum(r["bytes"] for r in evidence),
        "files": evidence,
    }
    if prior:
        assert prior == report
    else:
        save(RECEIPT, report)
    print(len(evidence), "original wheels,", report["total_bytes"], "physical bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download and independently measure the locked originals.",
    )
    verify(parser.parse_args().fetch)
