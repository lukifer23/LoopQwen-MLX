"""Exercise the installed research policy from a clean offline wheel outside the checkout."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.io import digest, read, save

POLICY = Path("research/results/observation-review-01")
QUESTION = (
    "A print shop has eight packs of 125 sheets. It prints 37 copies of a "
    "16-page booklet, using both sides of each sheet. How many sheets remain? "
    "Use the calculator and give the final number."
)


def main():
    revision = subprocess.check_output(
        ["git", "rev-parse", "acb9fd9200e9ff646f5af56791e5c84b8323302c^{commit}"], cwd=ROOT, text=True
    ).strip()
    out = ROOT / "exports/observation-review-runtime"
    out.mkdir(parents=True, exist_ok=True)
    wheel = build(out, revision)
    args = [
        "ask",
        "--checkpoint",
        str(ROOT / "exports/tool-language-packed-05"),
        "--session-policy",
        str(ROOT / POLICY),
        "--tool-format",
        "qwen",
        "--calculator",
        "--prompt",
        QUESTION,
        "--max-tokens",
        "384",
        "--max-turns",
        "3",
        "--trace",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        site = tmp / "site"
        assert digest(build(tmp / "repeat", revision)) == digest(wheel)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--target",
                str(site),
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        cmd = [sys.executable, "-I", "-c", CODE, str(site), *args]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        result = json.loads(
            subprocess.check_output([*cmd, "--compare-base"], cwd=tmp, env=env)
        )
        standalone = json.loads(subprocess.check_output(cmd, cwd=tmp, env=env))
        assert trace(result["runs"]["recurrent"][0]) == trace(standalone)
    dst = ROOT / "docs/measurements/observation-review-installed-cli.json"
    evidence = {
        "source_revision": revision,
        "wheel": wheel.name,
        "wheel_sha256": digest(wheel),
        "wheel_bytes": wheel.stat().st_size,
        "policy": digest(ROOT / POLICY / "checksums.json"),
        "question": QUESTION,
        "reference": 8 * 125 - 37 * (16 // 2),
        "result": result,
        "independent_candidate_trace": trace(standalone),
        "offline": True,
        "outside_checkout": True,
    }
    if dst.exists():
        old = read(dst)
        for field in (
            "source_revision",
            "wheel_sha256",
            "policy",
            "question",
            "reference",
            "independent_candidate_trace",
        ):
            assert old[field] == evidence[field]
        for side in ("base", "recurrent"):
            assert trace(result["runs"][side][0]) == trace(
                old["result"]["runs"][side][0]
            )
    else:
        save(dst, evidence)
    print(wheel, [(s, r[0]["answer"]) for s, r in result["runs"].items()])


if __name__ == "__main__":
    main()
