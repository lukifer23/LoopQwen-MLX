"""Rebuild and exercise the offline function runtime from its recorded source revision."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from loopqwen.io import read, digest

ROOT = Path(__file__).resolve().parents[1]
CODE = """import sys
from pathlib import Path
site=Path(sys.argv.pop(1)).resolve()
sys.path.insert(0,str(site))
from loopqwen.cli import main
main()
for name,module in list(sys.modules.items()):
    if name=='loopqwen' or name.startswith('loopqwen.'):
        files=[module.__file__] if getattr(module,'__file__',None) else list(module.__path__)
        assert files and all(Path(f).resolve().is_relative_to(site) for f in files), name
"""


def trace(row):
    keys = (
        "prompt_ids",
        "token_ids",
        "call",
        "calls",
        "observation",
        "observations",
        "kv_history",
    )
    return [
        row["answer"],
        row["status"],
        [[t.get(k) for k in keys] for t in row["turns"]],
    ]


def build(directory, revision):
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        archive = work / "source.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "-o",
                str(archive),
                revision,
                "pyproject.toml",
                "README.md",
                "LICENSE",
                "loopqwen",
            ],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter="data")
        subprocess.run(
            ["uv", "build", "--wheel", "--offline", "--out-dir", str(directory)],
            env=dict(os.environ, SOURCE_DATE_EPOCH="315532800"),
            cwd=work,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    return next(directory.glob("*.whl"))


def members(path):
    with zipfile.ZipFile(path) as z:
        return {k: z.read(k) for k in z.namelist()}


def verify(wheel, revision):
    receipt = read(ROOT / "docs/measurements/native-function-wheel.json")
    assert digest(wheel) == receipt["wheel_sha256"]
    sessions = read(ROOT / receipt["reference"])["observations"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        one = build(tmp / "one", revision)
        two = build(tmp / "two", revision)
        assert digest(one) == digest(two) and members(one) == members(wheel)
        site = tmp / "site"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--target",
                str(site),
                "--no-deps",
                str(one),
            ],
            check=True,
        )
        for session in sessions:
            args = session["arguments"].copy()
            args.remove("--compare-base")
            for opt in ("--checkpoint", "--tool-adapter"):
                idx = args.index(opt) + 1
                args[idx] = str((ROOT / args[idx]).resolve())
            if "--file" in args:
                rel = Path(args[args.index("--file") + 1])
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dst)
            actual = json.loads(
                subprocess.check_output(
                    [sys.executable, "-I", "-c", CODE, str(site), *args],
                    cwd=tmp,
                    env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                )
            )
            assert [trace(r) for r in actual["exchanges"]] == [
                trace(r) for r in session["result"]["runs"]["recurrent"]
            ]
        print(
            json.dumps(
                {
                    "source_revision": revision,
                    "build_epoch": 315532800,
                    "reproducible_sha256": digest(one),
                    "shipped_sha256": digest(wheel),
                    "identical_uncompressed_files": len(members(one)),
                    "replayed_exchanges": sum(
                        len(r["result"]["runs"]["recurrent"]) for r in sessions
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        default=ROOT / "exports/wheels/loopqwen-0.1.0-py3-none-any.whl",
    )
    parser.add_argument("--revision", default="9ac3135473302edc7dad4498015a37a285c73dfd")
    args = parser.parse_args()
    verify(args.wheel, args.revision)
