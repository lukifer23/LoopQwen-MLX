"""Rebuild and verify the wheel that exposes the original-archive reader and native input traces."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.io import digest, read, save
from scripts.verify_runtime_wheel import ROOT, build

REVISION = "b65ef6076e3d5dfe3220a55aa0278670759fa193"
CHILD = """
import json,sys
from pathlib import Path
def offline(name, args):
    if name in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('Network connection from isolated wheel verification')
sys.addaudithook(offline)
site,source,cache=(Path(arg).resolve() for arg in sys.argv[1:4])
sys.path.insert(0,str(site))
from loopqwen import wheel_tools
from loopqwen.tool_cli import add_arguments
import argparse,hashlib
parser=argparse.ArgumentParser()
add_arguments(parser,Path('.'))
assert parser.parse_args(['--prompt','Read this','--wheel','one.whl']).wheel == [Path('one.whl')]
items=json.loads(source.read_bytes())['cohorts']
reads=0
for split in items.values():
    for w in split:
        archive=cache / w['file']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==w['sha256']
        (tool,)=wheel_tools.tools([archive])
        for field,expected in w['headers'].items():
            assert tool.callback(archive.name,field) == {'values':expected}
            reads+=1
for name,mod in sys.modules.items():
    if name=='loopqwen' or name.startswith('loopqwen.'):
        origins=[mod.__file__] if getattr(mod,"__file__",None) else list(mod.__path__)
        assert origins and all(Path(x).resolve().is_relative_to(site) for x in origins)
print(json.dumps({'archives':sum(map(len,items.values())),'reads':reads}))
"""


def run():
    folder = ROOT / "exports/followup-runtime-01"
    folder.mkdir(parents=True, exist_ok=True)
    first = build(folder, REVISION)
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        second = build(work, REVISION)
        assert digest(first) == digest(second)
        site = work / "installed"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--no-deps",
                "--target",
                str(site),
                str(second),
            ],
            check=True,
        )
        cache = work / "hf"
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        checks = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    CHILD,
                    str(site),
                    str(ROOT / "research/sources/wheel-archives/enrollment.json"),
                    str(ROOT / "data/research_external/wheel-archives/wheels"),
                ],
                cwd=work,
                env=env,
            )
        )
        assert checks == {"archives": 60, "reads": 360} and not list(cache.rglob("*"))
    receipt = {
        "source_revision": REVISION,
        "archive": str(first.relative_to(ROOT)),
        "sha256": digest(first),
        "bytes": first.stat().st_size,
        "independent_builds": 2,
        "installed_offline": True,
        **checks,
    }
    path = ROOT / "docs/measurements/followup-runtime.json"
    if path.exists():
        assert read(path) == receipt
    else:
        save(path, receipt)
    print(receipt)


if __name__ == "__main__":
    run()
