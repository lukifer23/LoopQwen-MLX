"""Restore a self-contained research model from the public pinned Qwen snapshot."""

import argparse
import json
import mmap
import shutil
import struct
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SHARD = "model.safetensors-00001-of-00001.safetensors"
OFFICIAL = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
KIT = (
    Path(__file__).resolve().parents[1]
    / "research/results/middle-depth-576/restore.zip"
)
RECEIPT = KIT.with_name("transport.json")


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def header(stream):
    prefix = stream.read(8)
    body = stream.read(struct.unpack("<Q", prefix)[0])
    return prefix + body, json.loads(body)


def add(out, key, data):
    info = ZipInfo(key)
    info.compress_type = ZIP_DEFLATED
    out.writestr(info, data)


def pack(snapshot, models):
    shard = snapshot / SHARD
    if digest(shard) != OFFICIAL:
        raise ValueError("Wrong upstream Qwen shard")
    KIT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=KIT.parent) as tmp:
        path = Path(tmp) / KIT.name
        with (
            ZipFile(path, "w") as out,
            shard.open("rb") as official,
            (models / "base/model.safetensors").open("rb") as native,
        ):
            src_head, src = header(official)
            head, weights = header(native)
            add(out, "header", head)
            layout = []
            with (
                mmap.mmap(official.fileno(), 0, access=mmap.ACCESS_READ) as a,
                mmap.mmap(native.fileno(), 0, access=mmap.ACCESS_READ) as b,
            ):
                for name, m in sorted(
                    ((n, m) for n, m in weights.items() if n != "__metadata__"),
                    key=lambda v: v[1]["data_offsets"],
                ):
                    i, j = m["data_offsets"]
                    data = memoryview(b)[len(head) + i : len(head) + j]
                    upstream = src[name.replace("model.", "model.language_model.", 1)]
                    lo, hi = upstream["data_offsets"]
                    start = len(src_head) + lo
                    if (
                        hi - lo == j - i
                        and sha256(data).digest()
                        == sha256(memoryview(a)[start : start + hi - lo]).digest()
                    ):
                        layout.append([j - i, start])
                    else:
                        add(out, "literal/" + name, data)
                        layout.append([j - i, "literal/" + name])
                    del data
            members = {}
            blobs = set()
            for arm in ("base", "loop"):
                folder = models / arm
                files = {}
                for p in sorted(folder.rglob("*")):
                    if not p.is_file():
                        continue
                    if p.is_symlink():
                        raise ValueError("Input must be standalone")
                    key = str(p.relative_to(folder))
                    sha = digest(p)
                    if key == "model.safetensors":
                        origin = "layout"
                    elif (snapshot / key).is_file() and digest(snapshot / key) == sha:
                        origin = "snapshot"
                    else:
                        origin = "kit"
                        if sha not in blobs:
                            add(out, "files/" + sha, p.read_bytes())
                            blobs.add(sha)
                    files[key] = [origin, sha]
                members[arm] = files
            add(
                out,
                "index.json",
                json.dumps(
                    {"members": members, "layout": layout}, sort_keys=True
                ).encode(),
            )
        path.replace(KIT)
    RECEIPT.write_text(
        json.dumps(
            {
                "snapshot_revision": REVISION,
                "shard_sha256": OFFICIAL,
                "kit_sha256": digest(KIT),
                "kit_bytes": KIT.stat().st_size,
                "copied_upstream_tensor_bytes": sum(
                    n for n, p in layout if isinstance(p, int)
                ),
                "literal_tensor_bytes": sum(n for n, p in layout if isinstance(p, str)),
            },
            indent=2,
        )
        + "\n"
    )


def restore(snapshot, arm, target):
    if target.exists():
        raise FileExistsError(target)
    shard = snapshot / SHARD
    if (
        digest(shard) != OFFICIAL
        or digest(KIT) != json.loads(RECEIPT.read_text())["kit_sha256"]
    ):
        raise ValueError("Wrong upstream snapshot or research artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temp, ZipFile(KIT) as z:
        folder = Path(temp) / "model"
        index = json.loads(z.read("index.json"))
        for name, (origin, sha) in index["members"][arm].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(name)
            p = folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if origin == "layout":
                with shard.open("rb") as source, p.open("wb") as out:
                    out.write(z.read("header"))
                    for length, offset in index["layout"]:
                        if isinstance(offset, str):
                            data = z.read(offset)
                            if len(data) != length:
                                raise ValueError(offset)
                            out.write(data)
                        else:
                            source.seek(offset)
                            while length:
                                chunk = source.read(min(length, 4 * 2**20))
                                if not chunk:
                                    raise EOFError(shard)
                                out.write(chunk)
                                length -= len(chunk)
            elif origin == "snapshot":
                shutil.copyfile(snapshot / name, p)
            elif origin == "kit":
                with z.open("files/" + sha) as src, p.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise ValueError(origin)
            if digest(p) != sha:
                raise ValueError(f"Restored file differs: {name}")
        folder.rename(target)


def restore_full_fit(parent, target):
    """Restore the full-document checkpoint from a verified standalone parent."""
    import subprocess
    import sys

    repo = KIT.parents[3]
    manifest = repo / "docs/measurements/middle-depth/standalone"
    required = json.loads((manifest / "loop-files.json").read_text())["files"]
    complete = json.loads((manifest / "long-loop-files.json").read_text())["files"]
    delta = repo / "research/results/middle-depth-full-document/changes.zip"
    if target.exists():
        raise FileExistsError(target)
    if (
        any(p.is_symlink() for p in parent.rglob("*"))
        or {
            str(p.relative_to(parent)): digest(p)
            for p in parent.rglob("*")
            if p.is_file()
        }
        != required
    ):
        raise ValueError(
            "Full-document checkpoint needs its verified standalone parent"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = {name for name, sha in complete.items() if required.get(name) != sha}
    with TemporaryDirectory(dir=target.parent) as tmp, ZipFile(delta) as z:
        folder = Path(tmp) / "model"
        if set(z.namelist()) != changed:
            raise ValueError("Unexpected full-document file set")
        if sys.platform == "darwin":
            subprocess.run(["cp", "-cR", parent, folder], check=True)
        else:
            shutil.copytree(parent, folder)
        for name in changed:
            with z.open(name) as src, (folder / name).open("wb") as dest:
                shutil.copyfileobj(src, dest)
        if {
            str(p.relative_to(folder)): digest(p)
            for p in folder.rglob("*")
            if p.is_file()
        } != complete:
            raise ValueError("Restored full-document weights differ")
        folder.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("restore")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--arm", choices=("base", "loop"), required=True)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("restore-full-fit")
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("pack")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--models", default=Path("exports/standalone-middle-576"), type=Path)
    a = parser.parse_args()
    if a.action == "pack":
        pack(a.snapshot, a.models)
    elif a.action == "restore-full-fit":
        restore_full_fit(a.parent, a.out)
    else:
        restore(a.snapshot, a.arm, a.out)
