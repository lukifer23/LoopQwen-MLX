"""Pair actual application pins with published incompatible releases for authentic wheel requirements."""

import argparse
import gzip
import hashlib
import json
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.wheel_tools import headers
from scripts.wheel_archive_sources import CACHE, fetch, locked, previous_projects

DEST = Path("research/sources/dependency-decisions")
CONTEXT = Path(
    "research/sources/locked-context/4ba8468a0130d890328bb821786bb74bdca32298.lock.gz"
)
CUTOFF = "2026-11-14"


def versions(name, acquire):
    dest = DEST / "releases" / (name + ".json.gz")
    if not dest.exists():
        if not acquire:
            raise FileNotFoundError(dest)
        raw = fetch("https://pypi.org/pypi/" + name + "/json")
        items = []
        for v, files in json.loads(raw)["releases"].items():
            try:
                version = Version(v)
            except InvalidVersion:
                continue
            if version.is_prerelease or version.is_devrelease:
                continue
            candidates = [
                x
                for x in files
                if not x["yanked"] and x["upload_time_iso_8601"][:10] <= CUTOFF
            ]
            if candidates:
                x = min(candidates, key=lambda v: v["filename"])
                items.append(
                    [
                        v,
                        x["filename"],
                        x["digests"]["sha256"],
                        x["upload_time_iso_8601"],
                    ]
                )
        items.sort(key=lambda q: (Version(q[0]), q[0]))
        data = {
            "project": name,
            "as_of": CUTOFF,
            "pypi_reply_sha256": hashlib.sha256(raw).hexdigest(),
            "versions": items,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
        )
    return json.loads(gzip.decompress(dest.read_bytes()))["versions"]


def enrolled(acquire=False):
    prior = read("research/sources/wheel-archives/enrollment.json")["cohorts"]
    old = {w["package"] for items in prior.values() for w in items}
    blocked = old | previous_projects()
    held_raw = tomllib.loads(gzip.decompress(CONTEXT.read_bytes()).decode())["package"]
    held = []
    for p in held_raw:
        name = canonicalize_name(p["name"])
        if name in blocked:
            continue
        fs = sorted(
            [
                f
                for f in p.get("wheels", [])
                if f["url"].endswith("none-any.whl") and f["size"] <= 1024**2
            ],
            key=lambda f: f["url"],
        )
        if fs:
            f = fs[0]
            held.append(
                {
                    "package": name,
                    "file": unquote(Path(urlsplit(f["url"]).path).name),
                    "url": f["url"],
                    "sha256": f["hash"].removeprefix("sha256:"),
                }
            )
    out = {}
    train_raw = locked("train", False)
    for split, wheels, lock, need in (
        ("held", held, held_raw, 10),
        ("train", prior["train"], train_raw, 20),
    ):
        pins = {}
        for x in lock:
            if "version" in x:
                pins.setdefault(canonicalize_name(x["name"]), set()).add(x["version"])
        selections = []
        owners = set()
        for item in sorted(
            wheels, key=lambda w: fingerprint([771920, split, w["file"]])
        ):
            if item["package"] in owners:
                continue
            file = CACHE / "wheels" / item["file"]
            if not file.exists():
                if not acquire:
                    raise FileNotFoundError(file)
                raw = fetch(item["url"])
                assert hashlib.sha256(raw).hexdigest() == item["sha256"]
                file.write_bytes(raw)
            assert digest(file) == item["sha256"]
            h = headers(file)
            reqs = [Requirement(str(r)) for r in h.get_all("Requires-Dist", [])]
            options = []
            for r in reqs:
                name = canonicalize_name(r.name)
                if (
                    r.marker
                    or r.url
                    or not str(r.specifier)
                    or len(pins.get(name, [])) != 1
                ):
                    continue
                if (
                    sum(
                        canonicalize_name(q.name) == name and q.marker is None
                        for q in reqs
                    )
                    != 1
                ):
                    continue
                pin = next(iter(pins[name]))
                if not r.specifier.contains(pin):
                    continue
                options.append((fingerprint([771921, str(r)]), r, name, pin))
            for _, r, name, pin in sorted(options, key=lambda v: v[0]):
                releases = versions(name, acquire)
                eligible = [
                    (j, x)
                    for j, x in enumerate(releases)
                    if Version(x[0]) == Version(pin)
                ]
                if not eligible:
                    continue
                index, allowed = eligible[0]
                denied = [
                    (abs(j - index), j, x)
                    for j, x in enumerate(releases)
                    if not r.specifier.contains(x[0])
                ]
                if not denied:
                    continue
                wrong = min(denied)[2]
                with ZipFile(file) as archive:
                    member = next(
                        n
                        for n in archive.namelist()
                        if n.endswith(".dist-info/METADATA")
                    )
                    mh = hashlib.sha256(archive.read(member)).hexdigest()
                selections.append(
                    {
                        **{k: item[k] for k in ("package", "file", "url", "sha256")},
                        "metadata_sha256": mh,
                        "dependency": name,
                        "requirement": str(r.specifier),
                        "header": str(r),
                        "allowed_release": allowed,
                        "denied_release": wrong,
                    }
                )
                owners.add(item["package"])
                break
            if len(selections) == need:
                break
        if len(selections) != need:
            raise ValueError(f"Only {len(selections)}/{need} eligible {split} projects")
        out[split] = selections
    result = {
        "source_lock_sha256": digest(CONTEXT),
        "cutoff_utc": CUTOFF,
        "cohorts": out,
    }
    path = DEST / "originals.json"
    if path.exists():
        assert read(path) == result
    else:
        save(path, result)
    print({k: len(v) for k, v in out.items()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    enrolled(parser.parse_args().acquire)
