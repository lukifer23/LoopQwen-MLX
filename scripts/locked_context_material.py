"""Verify two adjacent published lockfiles and reconstruct the complete ordinary chat inputs."""

import argparse
import gzip
import hashlib
import re
import tomllib
from pathlib import Path
from urllib.request import urlopen

from loopqwen.file_tools import definitions
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.qwen_tools import declarations, render
from loopqwen.research.cross_view_study import tokenizer

SOURCE = Path("research/sources/locked-context")
MATERIAL = Path("research/data/locked-context.json")
ORIGIN = "https://raw.githubusercontent.com/python-jsonschema/jsonschema"
VERSIONS = [
    (
        "648f1ccd472451e31f56a0a6f5596d71313a9ffa",
        "72f0c098f10712d65ab0855c5b038069518a65cd",
    ),
    (
        "4ba8468a0130d890328bb821786bb74bdca32298",
        "795385ed4932e1119d9e32e2ad0e5d337f3f1368",
    ),
]
QUESTION = (
    'Which exact version of the package "virtue" is pinned in this uv.lock? '
    "Use the file tool to measure uv.lock in bytes. Give the pinned version and the byte count."
)


def request(source):
    return (
        "Here is the complete uv.lock file.\n\n"
        + source
        + "\nEnd of uv.lock.\n\n"
        + QUESTION
    )


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def material(acquire=False):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    catalogue = declarations(definitions(None, ["uv.lock"]))
    rows = []
    pins = []
    assert (
        blob((SOURCE / "COPYING").read_bytes())
        == "af9cfbdb134f42e5205ecbad597421d778826481"
    )
    for ref, expected in VERSIONS:
        path = SOURCE / (ref + ".lock.gz")
        if acquire:
            raw = urlopen(f"{ORIGIN}/{ref}/uv.lock", timeout=30).read()
            assert blob(raw) == expected
            path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        raw = gzip.decompress(path.read_bytes())
        assert blob(raw) == expected
        source = raw.decode()
        packages = tomllib.loads(source)["package"]
        versions = {}
        for p in packages:
            versions.setdefault(p["name"], set()).add(p.get("version"))
        pins.append(versions)
        (version,) = versions["virtue"]
        user = request(source)
        chat = render(tok, [{"role": "user", "content": user}], catalogue)
        ids = tok.encode(chat, add_special_tokens=False)
        location = re.search(r'(?m)^name = "virtue"\nversion = "', source)
        assert location and source[location.end() :].startswith(version + '"')
        at = chat.index(source) + location.end()
        assert len(ids) >= 131072
        rows.append(
            {
                "revision": ref,
                "git_blob": expected,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "url": f"{ORIGIN}/{ref}/uv.lock",
                "bytes": len(raw),
                "packages": len(packages),
                "version": version,
                "version_token_offset": len(
                    tok.encode(chat[:at], add_special_tokens=False)
                ),
                "prompt_tokens": len(ids),
                "prompt_sha256": fingerprint(ids),
            }
        )
    assert {n for n in pins[0].keys() & pins[1].keys() if pins[0][n] != pins[1][n]} == {
        "virtue"
    }
    return {
        "upstream": "python-jsonschema/jsonschema",
        "path": "uv.lock",
        "question": QUESTION,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--write", action="store_true")
    opts = parser.parse_args()
    data = material(opts.acquire)
    if opts.write:
        save(MATERIAL, data)
    assert data == read(MATERIAL)
    for r in data["rows"]:
        print(
            r["revision"][:9],
            r["bytes"],
            r["version"],
            r["prompt_tokens"],
            r["version_token_offset"],
        )
