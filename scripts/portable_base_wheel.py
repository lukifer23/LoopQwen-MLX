"""Replay the official dense control from one release, with an empty model cache and no network."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

REVISION = "91b71379662baa0113f3878e239d1ecc70f45cfa"
NETWORK_GUARD = """import sys
def isolated(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise AssertionError('Network access attempted by offline baseline')
sys.addaudithook(isolated)
"""


def main():
    docs = ROOT / "docs/measurements"
    directory = ROOT / "exports/portable-base-runtime"
    directory.mkdir(parents=True, exist_ok=True)
    wheel = build(directory, REVISION)
    bundle = ROOT / "exports/tool-language-packed-05"
    native = read(docs / "native-function-installed-cli.json")["observations"][0]
    raw = read(docs / "source-return-installed-cli.json")["requests"]
    sessions = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        site = tmp / "site"
        second = build(tmp / "repeat", REVISION)
        assert digest(wheel) == digest(second)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
        )
        cache = tmp / "empty-model-cache"
        cache.mkdir()

        def execute(args):
            cmd = [sys.executable, "-I", "-c", NETWORK_GUARD + CODE, str(site), *args]
            env = dict(
                os.environ,
                HF_HOME=str(cache),
                HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
                HF_HUB_CACHE=str(cache / "hub"),
                TRANSFORMERS_CACHE=str(cache / "hub"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            r = json.loads(subprocess.check_output(cmd, cwd=tmp, env=env))
            assert (
                r["base_source"] == "certified_release_text"
                and r["initial_prompt_identical"]
            )
            return r

        args = native["arguments"].copy()
        for opt in ("--checkpoint", "--tool-adapter"):
            i = args.index(opt) + 1
            args[i] = str(ROOT / args[i])
        actual = execute(args)
        for side in ("recurrent", "base"):
            assert [trace(r) for r in actual["runs"][side]] == [
                trace(r) for r in native["result"]["runs"][side]
            ]
        sessions.append(
            {
                "kind": "functions",
                "source": "native-function-installed-cli.json",
                "index": 0,
                "identical_trajectories": {
                    s: len(v) for s, v in actual["runs"].items()
                },
                "base_trace_sha256": fingerprint(
                    [trace(r) for r in actual["runs"]["base"]]
                ),
            }
        )
        for i, obs in enumerate(raw):
            args = obs["arguments"].copy()
            args[args.index("--checkpoint") + 1] = str(bundle)
            actual = execute(args)
            a, b = actual["runs"]["base"], obs["result"]["runs"]["base"]
            for key in ("token_ids", "text", "initial_sha256", "prompt_tokens"):
                assert a[key] == b[key]
            sessions.append(
                {
                    "kind": "prose",
                    "source": "source-return-installed-cli.json",
                    "index": i,
                    "initial_sha256": a["initial_sha256"],
                    "tokens": len(a["token_ids"]),
                    "output_sha256": fingerprint(a["token_ids"]),
                }
            )
        assert not any(
            p.is_file()
            and (p.suffix in {".safetensors", ".bin"} or "snapshots" in p.parts)
            for p in cache.rglob("*")
        )
    evidence = {
        "revision": REVISION,
        "wheel": wheel.name,
        "sha256": digest(wheel),
        "bytes": wheel.stat().st_size,
        "source_inventory": digest(bundle / "tool-export-files.json"),
        "outside_checkout": True,
        "empty_model_cache": True,
        "network_connections_blocked": True,
        "sessions": sessions,
    }
    path = docs / "portable-base-wheel.json"
    if path.exists():
        assert read(path) == evidence
    else:
        save(path, evidence)
    print(wheel, evidence["bytes"], evidence["sha256"])


if __name__ == "__main__":
    main()
