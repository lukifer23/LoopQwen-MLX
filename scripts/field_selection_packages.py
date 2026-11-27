"""Acquire and cross-check PyPI release metadata against the public simple index."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name

names = """arrow 1.3.0
anytree 2.12.1
black 25.1.0
rich 13.9.4
pydantic 2.11.7
flask 3.1.1
jinja2 3.1.6
werkzeug 3.1.3
typer 0.16.0
nltk 3.9.1
pandas 2.2.3
scipy 1.15.3
networkx 3.4.2
beautifulsoup4 4.13.4
sqlalchemy 2.0.41
alembic 1.16.1
babel 2.17.0
tabulate 0.9.0
zipp 3.21.0
openpyxl 3.1.5
tenacity 9.1.2
tornado 6.5.1
more-itertools 10.7.0
pexpect 4.9.0
mypy 1.16.0
pytz 2025.2
pympler 1.1
tomlkit 0.13.3
wheel 0.45.1
pygments 2.19.1""".splitlines()
old = (
    json.loads(Path("research/data/reasoned-evidence-panel.json").read_bytes())[
        "wheels"
    ]
    + json.loads(Path("research/data/proof-curriculum-training.json").read_bytes())[
        "package"
    ]
)
excluded = {canonicalize_name(w["package"]) for w in old}
dest = Path("research/sources/field-selection/metadata")
dest.mkdir(parents=True, exist_ok=True)


def get(link, accept="application/json"):
    req = urllib.request.Request(
        link,
        headers={
            "User-Agent": "LoopQwen-MLX public research provenance",
            "Accept": accept,
        },
    )
    with urllib.request.urlopen(req, timeout=22) as src:
        return src.read()


def fetch(pair):
    name, ver = pair.split()
    if name in excluded:
        return
    link = f"https://pypi.org/pypi/{name}/{ver}/json"
    d = json.loads(get(link))
    ws = sorted(
        (w for w in d["urls"] if w["filename"].endswith(".whl")),
        key=lambda w: (
            not w["filename"].endswith("py3-none-any.whl"),
            not ("macosx" in w["filename"] and "arm64" in w["filename"]),
            w["filename"],
        ),
    )
    if not ws:
        return
    wheel = ws[0]
    url = wheel["url"] + ".metadata"
    raw = get(url)
    head = BytesParser().parsebytes(raw)
    spec = head["Requires-Python"]
    if not spec:
        return
    assert (
        canonicalize_name(head["Name"]) == name
        and head["Version"] == ver
        and spec.replace(" ", "") == d["info"]["requires_python"].replace(" ", "")
    )
    simple = json.loads(
        get(f"https://pypi.org/simple/{name}/", "application/vnd.pypi.simple.v1+json")
    )
    posted = next(f for f in simple["files"] if f["filename"] == wheel["filename"])
    assert (
        posted["hashes"]["sha256"] == wheel["digests"]["sha256"]
        and posted["core-metadata"]["sha256"] == hashlib.sha256(raw).hexdigest()
    )
    path = dest / (name + ".metadata")
    path.write_bytes(raw)
    return {
        "package": head["Name"],
        "version": ver,
        "requirement": spec,
        "wheel": wheel["filename"],
        "release_url": link,
        "wheel_url": wheel["url"],
        "wheel_sha256": wheel["digests"]["sha256"],
        "metadata_url": url,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
    }


if __name__ == "__main__":
    with ThreadPoolExecutor(6) as pool:
        rows = [r for r in pool.map(fetch, names) if r]
    Path("research/sources/field-selection/inventory.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(len(rows), [(r["package"], r["requirement"]) for r in rows])
