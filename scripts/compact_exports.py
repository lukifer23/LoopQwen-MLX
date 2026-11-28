"""Audit equal-layout local exports. --apply shares unchanged extents on APFS."""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from collections import defaultdict


def metadata(p):
    s = p.stat()
    return (
        s.st_mode,
        s.st_uid,
        s.st_gid,
        s.st_flags,
        subprocess.check_output(["/usr/bin/xattr", "-lx", str(p)]),
    )


def compare(base, original, clone=None):
    digest = hashlib.sha256()
    same = 0
    while block := original.read(256 * 1024):
        donor = base.read(len(block))
        if donor == block:
            same += len(block)
        elif clone:
            clone.seek(original.tell() - len(block))
            clone.write(block)
        digest.update(block)
    return same, digest.hexdigest()


def compact(base, path, apply):
    before = path.stat()
    if before.st_nlink != 1 or metadata(base) != metadata(path):
        return 0
    with base.open("rb") as a, path.open("rb") as b:
        same, expected = compare(a, b)
    if apply and same > before.st_size / 2:
        # The temp and old target are independent files. Replace only after a full rehash.
        with tempfile.TemporaryDirectory(dir=path.parent) as work:
            tmp = Path(work) / path.name
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.clonefile(os.fsencode(base), os.fsencode(tmp), 0):
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            with base.open("rb") as a, path.open("rb") as b, tmp.open("r+b") as c:
                compare(a, b, c)
                c.flush()
                os.fsync(c.fileno())
            with tmp.open("rb") as t:
                if hashlib.file_digest(t, "sha256").hexdigest() != expected:
                    raise ValueError(f"Byte verification failed for {path}")
            now = path.stat()
            if (now.st_ino, now.st_ctime_ns) != (before.st_ino, before.st_ctime_ns):
                raise RuntimeError(f"Source changed during compression: {path}")
            os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(tmp, path)
    print(f"{path}  equal blocks {same / 2**20:.1f} MiB  sha256 {expected}", flush=True)
    return same


def compact_indexed_features(apply):
    """Share complete, byte-identical legacy position shards listed in saved indexes."""
    import gzip
    import json
    from loopqwen.io import digest, read, save

    groups = defaultdict(set)
    indexes = {}
    for index in sorted(Path("checkpoints/research").glob("*/partial-index.json.gz")):
        with gzip.open(index, "rt") as stream:
            records = json.load(stream)
        if not isinstance(records, list) or not records or "sha256" not in records[0]:
            continue
        folder = index.parent / "positions"
        if not folder.is_dir() or folder.is_symlink():
            continue
        for r in records:
            name = r["file"]
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError(f"Unexpected indexed feature path in {index}")
            p = folder / name
            if p.is_symlink() or not p.is_file():
                raise ValueError(f"Incomplete indexed feature: {p}")
            groups[r["sha256"]].add(p)
        indexes[str(index)] = digest(index)

    receipt = Path("docs/measurements/indexed-feature-storage.json")
    prior = read(receipt)["clones"] if receipt.exists() else {}
    clones = {}
    equal_bytes = 0
    for sha, paths in groups.items():
        if len(paths) < 2:
            continue
        donor, *copies = sorted(paths)
        if digest(donor) != sha:
            raise ValueError(f"Indexed source changed: {donor}")
        for dest in copies:
            if digest(dest) != sha:
                raise ValueError(f"Indexed target changed: {dest}")
            if dest.samefile(donor):
                continue
            n = dest.stat().st_size
            equal_bytes += n
            rec = prior.get(str(dest), {})
            if (
                rec.get("sha256") == sha
                and rec.get("source") == str(donor)
                and rec.get("source_inode") == donor.stat().st_ino
                and rec.get("target_inode") == dest.stat().st_ino
                and rec.get("target_ctime_ns") == dest.stat().st_ctime_ns
            ):
                clones[str(dest)] = rec
            elif apply and compact(donor, dest, True) == n:
                if digest(dest) != sha or dest.samefile(donor):
                    raise ValueError("Independent clone verification failed")
                clones[str(dest)] = {
                    "source": str(donor),
                    "bytes": n,
                    "sha256": sha,
                    "source_inode": donor.stat().st_ino,
                    "target_inode": dest.stat().st_ino,
                    "target_ctime_ns": dest.stat().st_ctime_ns,
                }
    verified = sum(r["bytes"] for r in clones.values())
    if apply:
        save(
            receipt,
            {
                "indexes": indexes,
                "indexed_paths": sum(map(len, groups.values())),
                "unique_indexed_hashes": len(groups),
                "verified_equal_payload_bytes": equal_bytes,
                "verified_cloned_payload_bytes": verified,
                "clones": clones,
                "measurement": "Matching indexed file bytes with independently writable APFS clones. Snapshots, allocation and pre-existing extent sharing affect physical free space.",
            },
        )
    print(
        len(clones),
        "verified clones;",
        verified,
        "cloned bytes;",
        equal_bytes,
        "equal payload bytes",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="atomically repack verified local exports"
    )
    parser.add_argument(
        "--indexed-features",
        action="store_true",
        help="audit complete duplicate position shards in saved research indexes",
    )
    args = parser.parse_args()
    if args.indexed_features:
        compact_indexed_features(args.apply)
        return
    groups = defaultdict(list)
    for path in sorted(Path("exports").glob("*/*")):
        if (
            path.name
            in {"model.safetensors", "tokenizer.json", "merges.txt", "vocab.json"}
            and path.is_file()
            and not path.is_symlink()
            and not path.parent.is_symlink()
        ):
            groups[path.name, path.stat().st_size].append(path)
    total = sum(compact(g[0], p, args.apply) for g in groups.values() for p in g[1:])
    print(
        f"Matching contents across candidate files: {total / 2**30:.3f} GiB. "
        "Existing filesystem sharing and snapshots affect physical savings."
    )


if __name__ == "__main__":
    main()
