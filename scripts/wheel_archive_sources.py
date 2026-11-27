"""Enroll original wheel archives from two independently versioned application lockfiles."""

import argparse
import hashlib
import tomllib
import urllib.request
from importlib.metadata import Distribution
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.wheel_tools import tools

ROOT = Path("research/sources/wheel-archives")
CACHE = Path("data/research_external/wheel-archives")
LOCKS = {
    "train": (
        "pydantic/pydantic",
        "6449f80db0214a95178b6b80a3613451b924a4f1",
        "87d563f01d750427776119145953e9184d6388ab",
    ),
    "held": (
        "fastapi/fastapi",
        "05e6df7e906c3decba377c37548019973f40bf15",
        "c4f862e8673f45f06031236dd91c02b3dd1b7b0a",
    ),
}
PYTHONS = (
    "3.6.15",
    "3.7.17",
    "3.8.18",
    "3.9.18",
    "3.10.13",
    "3.11.8",
    "3.12.2",
    "3.13.0",
    "3.14.0",
)
FIELDS = (
    "Name",
    "Version",
    "Requires-Python",
    "Summary",
    "Requires-Dist",
    "Provides-Extra",
)


def fetch(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "LoopQwen-MLX source provenance"}
    )
    with urllib.request.urlopen(req, timeout=35) as r:
        return r.read()


def locked(split, acquire):
    project, rev, expected = LOCKS[split]
    path = CACHE / (split + ".lock")
    if not path.exists() and acquire:
        path.write_bytes(
            fetch(f"https://raw.githubusercontent.com/{project}/{rev}/uv.lock")
        )
    raw = path.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()
        == expected
    )
    return tomllib.loads(raw.decode())["package"]


def candidates(split, acquire):
    out = {}
    for p in locked(split, acquire):
        name = canonicalize_name(p["name"])
        for f in p.get("wheels", []):
            file = unquote(Path(urlsplit(f["url"]).path).name)
            if file.endswith("-none-any.whl") and f["size"] <= 1024**2:
                key = (name, p["version"])
                q = {
                    "package": name,
                    "version": p["version"],
                    "file": file,
                    "url": f["url"],
                    "sha256": f["hash"].removeprefix("sha256:"),
                    "bytes": f["size"],
                }
                if key not in out or file < out[key]["file"]:
                    out[key] = q
    return sorted(out.values(), key=lambda x: fingerprint([740031, split, x["file"]]))


def original(item, acquire):
    dest = CACHE / "wheels" / item["file"]
    if not dest.exists() and acquire:
        raw = fetch(item["url"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        dest.write_bytes(raw)
    assert dest.stat().st_size == item["bytes"] and digest(dest) == item["sha256"]
    with ZipFile(dest) as z:
        (key,) = [
            p
            for p in z.namelist()
            if p.count("/") == 1 and p.endswith(".dist-info/METADATA")
        ]
        metadata = z.read(key)
    (tool,) = tools([dest])
    (dist,) = Distribution.discover(path=[str(dest)])
    values = {k: tool.callback(dest.name, k)["values"] for k in FIELDS}
    assert values == {k: list(map(str, dist.metadata.get_all(k, []))) for k in FIELDS}
    assert canonicalize_name(values["Name"][0]) == item["package"] and Version(
        values["Version"][0]
    ) == Version(item["version"])
    req = values["Requires-Python"]
    if len(req) != 1 or not req[0].strip():
        return None
    spec = SpecifierSet(req[0])
    allowed = [spec.contains(v) for v in PYTHONS]
    if all(allowed) or not any(allowed):
        return None
    # Prefer neighboring, actually released interpreters on different sides.
    boundary = next(i for i in range(1, len(PYTHONS)) if allowed[i] != allowed[i - 1])
    return {
        **item,
        "metadata_member": key,
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "headers": values,
        "interpreters": [[v, a] for v, a in zip(PYTHONS, allowed, strict=True)],
        "boundary": [PYTHONS[boundary - 1], PYTHONS[boundary]],
    }


def previous_projects():
    names = {
        canonicalize_name(r["name"])
        for r in tomllib.loads(Path("uv.lock").read_text())["package"]
    }
    for name in (
        "field-selection-panel",
        "valid-return-panel",
        "reasoned-evidence-panel",
        "proof-curriculum-training",
        "valid-return-training",
    ):
        d = read("research/data/" + name + ".json")
        for k in ("wheels", "package", "packages"):
            names.update(
                canonicalize_name(w["package"]) for w in d.get(k, []) if "package" in w
            )
    names.update(
        canonicalize_name(r["package"])
        for r in read("research/sources/field-selection/inventory.json")
    )
    return names


def run(acquire=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "wheels").mkdir(exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "enrollment.json"
    blocked = previous_projects()
    result = {}
    # Reserve fresh project names for the held application before drawing any training examples.
    for split, n in (("held", 12), ("train", 48)):
        seen = set()
        records = []
        for w in candidates(split, acquire):
            if w["package"] in seen or w["package"] in blocked:
                continue
            item = original(w, acquire)
            if item is None:
                continue
            seen.add(w["package"])
            records.append(item)
            if len(records) == n:
                break
        if len(records) != n:
            raise ValueError(
                f"Only {len(records)} eligible {split} projects for {n} planned"
            )
        result[split] = records
        blocked = seen if split == "held" else blocked | seen
    enrollment = {
        "source_locks": {k: list(v) for k, v in LOCKS.items()},
        "ordering_seed": 740031,
        "limits": {"max_wheel_bytes": 1024**2},
        "support_versions": list(PYTHONS),
        "cohorts": result,
    }
    if frozen.exists():
        assert read(frozen) == enrollment
    else:
        save(frozen, enrollment)
    print("Enrolled", [(k, len(v)) for k, v in result.items()])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    run(p.parse_args().acquire)
