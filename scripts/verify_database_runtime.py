"""Build twice, install offline, then replay the original database queries through the wheel."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.io import digest, read, save
from scripts.dialogue_database_sources import CACHE, DEST
from scripts.verify_runtime_wheel import ROOT, build

REVISION = "2a86ecb575dc46ff5a83d886ed5593d312630f27"
CHILD = """
import argparse, json, runpy, sys
from pathlib import Path
site, oracle, source, archive = (Path(x).resolve() for x in sys.argv[1:])
def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError(event)
sys.addaudithook(offline)
sys.path.insert(0, str(site))
from loopqwen.tool_cli import add_arguments
parser = argparse.ArgumentParser()
add_arguments(parser, Path('.'))
args = parser.parse_args(['--prompt', 'Read the catalog', '--relational-db', 'first.db',
                          '--relational-db', 'second.db'])
assert args.relational_db == [Path('first.db'), Path('second.db')]
result = runpy.run_path(str(oracle))['replay'](source, archive)
for name, mod in list(sys.modules.items()):
    if name == 'loopqwen' or name.startswith('loopqwen.'):
        paths = [mod.__file__] if getattr(mod, '__file__', None) else list(mod.__path__)
        assert paths and all(Path(p).resolve().is_relative_to(site) for p in paths), name
print(json.dumps(result))
"""


def run():
    dest = ROOT / "exports/database-runtime-01"
    dest.mkdir(parents=True, exist_ok=True)
    wheel = build(dest, REVISION)
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        independent = build(work, REVISION)
        assert digest(independent) == digest(wheel)
        site = work / "site"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--no-index",
                "--offline",
                "--no-deps",
                "--target",
                str(site),
                str(independent),
            ],
            check=True,
        )
        oracle = ROOT / "scripts/dialogue_database_oracle.py"
        result = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    CHILD,
                    str(site),
                    str(oracle),
                    str(ROOT / DEST / "originals.json"),
                    str(ROOT / CACHE),
                ],
                cwd=work,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
            )
        )
    measured = ROOT / "docs/measurements/database-dialogue-tools.json"
    assert result == read(measured)
    receipt = {
        "source_revision": REVISION,
        "wheel": str(wheel.relative_to(ROOT)),
        "wheel_sha256": digest(wheel),
        "bytes": wheel.stat().st_size,
        "independent_builds": 2,
        "isolated_offline_install": True,
        "query_receipt_sha256": digest(measured),
        "oracle_sha256": digest(oracle),
        **{
            k: result[k]
            for k in ("databases", "queries", "referencing_multiple_tables")
        },
    }
    path = ROOT / "docs/measurements/database-runtime.json"
    if path.is_file():
        assert read(path) == receipt
    else:
        save(path, receipt)
    print(receipt)


if __name__ == "__main__":
    run()
