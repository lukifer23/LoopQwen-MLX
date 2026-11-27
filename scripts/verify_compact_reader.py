"""Replay both versions of the first frozen article using an isolated wheel and no network."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, build

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

REV = "79896bbee8df16c0d99592771e91af5f2404149b"


def main():
    spec = read(ROOT / "research/experiments/revision-evidence-01.json")
    study = ROOT / "research/results" / spec["id"]
    reader = ROOT / spec["export"]
    release = ROOT / spec["release"]
    panel = read(ROOT / "research/data/revision-evidence-01.json")
    dest = ROOT / "exports/compact-revision-runtime-01"
    dest.mkdir(parents=True, exist_ok=True)
    wheel = build(dest, REV)
    out = []
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        assert digest(build(work / "repeat", REV)) == digest(wheel)
        envdir = work / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--offline",
                "--no-project",
                "--python",
                sys.executable,
                str(envdir),
            ],
            check=True,
        )
        python = str(envdir / "bin/python")
        subprocess.run(
            [
                "uv",
                "--no-cache",
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--python",
                python,
                "--find-links",
                str(ROOT / "exports/field-dependencies-01"),
                str(wheel),
            ],
            check=True,
        )
        subprocess.run(["uv", "pip", "check", "--python", python], check=True)
        site = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
        cache = work / "empty-hf"
        cache.mkdir()
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
            TRANSFORMERS_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        for i, row in enumerate(panel):
            if row["case"] != panel[0]["case"]:
                continue
            args = [
                "generate",
                "--checkpoint",
                str(reader),
                "--compare-base",
                "--base-checkpoint",
                str(release),
                "--thinking",
                "--trace",
                "--prompt",
                row["body"],
                "--max-tokens",
                str(spec["generation_cap"]),
            ]
            actual = json.loads(
                subprocess.check_output(
                    [python, "-I", "-c", NETWORK_GUARD + CODE, site, *args],
                    cwd=work,
                    env=env,
                )
            )
            assert (
                actual["base_source"] == "certified_release_text"
                and actual["initial_prompt_identical"]
            )
            identities = {}
            for side, arm in (("recurrent", "candidate"), ("base", "base")):
                reference = unpack(study / (arm + ".json.gz"))[i]
                result = actual["runs"][side]
                assert reference["id"] == row["id"] and result[
                    "initial_sha256"
                ] == fingerprint(row["prompt_ids"])
                assert all(
                    result[k] == reference[k]
                    for k in ("token_ids", "kv_history", "text", "prompt_tokens")
                )
                identities[side] = {
                    "tokens": len(result["token_ids"]),
                    "token_sha256": fingerprint(result["token_ids"]),
                    "attention_histories": result["kv_history"],
                    "visible": reference["score"]["visible"],
                    "correct": reference["score"]["correct"],
                }
            out.append(
                {
                    "source_id": row["id"],
                    "label": row["label"],
                    "initial_ids_sha256": fingerprint(row["prompt_ids"]),
                    "runs": identities,
                }
            )
        assert len(out) == 2 and not any(x.is_file() for x in cache.rglob("*"))
    weightfile = reader / "model.safetensors"
    assert (
        digest(weightfile)
        == read(study / "compact-entailment.json")["files"][weightfile.name]
    )
    receipt = {
        "wheel_source": REV,
        "wheel": str(wheel.relative_to(ROOT)),
        "wheel_sha256": digest(wheel),
        "weight_bytes": weightfile.stat().st_size,
        "weight_sha256": digest(weightfile),
        "empty_model_cache": True,
        "fresh_venv": True,
        "network_blocked": True,
        "examples": out,
    }
    dest = ROOT / "docs/measurements/compact-revision-runtime.json"
    if dest.exists():
        assert read(dest) == receipt
    else:
        save(dest, receipt)
    print(
        "Two complete article snapshots replayed on both native models from the installed wheel"
    )


if __name__ == "__main__":
    main()
