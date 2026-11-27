"""Collect the exact runtime wheels from uv.lock for the host Apple Silicon Python."""

import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from loopqwen.io import digest

ROOT = Path(__file__).resolve().parents[1]


def wheelhouse(directory, *, only_group=None):
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("Build this wheelhouse with macOS arm64 Python 3.12.")
    packages = {
        p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    }
    requirements = subprocess.check_output(
        [
            "uv",
            "export",
            "--frozen",
            "--offline",
            *(["--only-group", only_group] if only_group else ["--no-dev"]),
            "--no-emit-project",
            "--no-header",
            "--no-hashes",
            "--no-annotate",
            "--no-editable",
        ],
        cwd=ROOT,
        text=True,
    )
    priority = {t: i for i, t in enumerate(sys_tags())}
    directory.mkdir(exist_ok=True, parents=True)
    paths = []
    for r in map(Requirement, requirements.splitlines()):
        if r.marker and not r.marker.evaluate():
            continue
        p = packages[r.name]
        assert r.specifier.contains(p["version"])
        matches = []
        for wheel in p["wheels"]:
            filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[1])
            tags = parse_wheel_filename(filename)[3]
            ranks = [priority[t] for t in tags if t in priority]
            if ranks:
                matches.append((min(ranks), filename, wheel))
        if not matches:
            raise ValueError(f"No compatible locked wheel for {r}")
        _, filename, wheel = min(matches, key=lambda choice: choice[:2])
        target = directory / filename
        expected = wheel["hash"].removeprefix("sha256:")
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=directory) as tmp:
                staged = Path(tmp) / filename
                with (
                    urlopen(wheel["url"], timeout=60) as response,
                    staged.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out)
                if digest(staged) != expected:
                    raise ValueError(
                        f"Downloaded wheel differs from uv.lock: {filename}"
                    )
                staged.replace(target)
        if target.stat().st_size != wheel["size"] or digest(target) != expected:
            raise ValueError(f"Local wheel differs from uv.lock: {filename}")
        paths.append(target)
    return paths


if __name__ == "__main__":
    wheels = wheelhouse(ROOT / "exports/field-dependencies-01")
    print(
        len(wheels),
        "locked runtime wheels,",
        sum(w.stat().st_size for w in wheels),
        "bytes",
    )
