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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="atomically repack verified local exports"
    )
    args = parser.parse_args()
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
