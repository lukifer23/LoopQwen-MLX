"""Restore and verify the exact published bytes needed by the long MLX experiment."""

import argparse
import gzip
import hashlib
from pathlib import Path
from urllib.request import urlopen

from loopqwen.io import digest, read
from scripts.agenda_sources import source
from scripts.book_conversation_material import SOURCES
from scripts.middle_depth_documents import SPECS, FROZEN


def restore(fetch):
    dev = read(FROZEN)
    gables = read(SPECS / "long-fit.json")["rows"]["gables/published"]
    anne = read(SPECS / "anne-dev.json")
    items = [
        (Path(read(SPECS / "book-dev.json")["source"]), dev["book"]["source"], None),
        (Path(anne["source"]), dev["anne"]["source"], anne["url"]),
        (
            Path(gables["source"]),
            gables["source_sha"],
            "https://raw.githubusercontent.com/GITenberg/" + SOURCES[0][1],
        ),
    ]
    for path, sha, url in items:
        if not path.exists() and fetch:
            if url is None:
                content = gzip.decompress(
                    Path(
                        "research/sources/middle-depth/mcteague-edition.txt.gz"
                    ).read_bytes()
                )
            else:
                with urlopen(url, timeout=60) as response:
                    content = response.read()
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError(f"Different edition supplied at {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".restoring")
            tmp.write_bytes(content)
            tmp.replace(path)
        if digest(path) != sha:
            raise ValueError(f"Unexpected edition at {path}")
    for fmt in ("xml", "ical"):
        source(2025, fmt, fetch)
    print("Verified three complete books and both publisher schedule files.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    restore(p.parse_args().restore)
