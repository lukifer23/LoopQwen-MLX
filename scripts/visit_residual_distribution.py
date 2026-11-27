"""Ship the three learned matrices and replay genuine request pairs from a clean installation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_kit import write_archive
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save


def run():
    archive = ROOT / "research/results/visit-residual-01"
    spec = read(archive / "registration.json")
    revision = read(archive / "evaluation-restart.json")["loader_revision"]
    source = read(archive / "checksums.json")
    panel = read(ROOT / spec["material"])["rows"]
    selected = []
    for family in ("effects", "dependency", "same_version"):
        selected.append(
            next((i, w) for i, w in enumerate(panel) if w["family"] == family)
        )
    output = ROOT / "exports/loopqwen-visit-policy-01.zip"
    receipt = ROOT / "docs/measurements/visit-residual-distribution.json"
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        wheel = build(work / "first", revision)
        assert digest(wheel) == digest(build(work / "second", revision))
        files = {
            "runtime/" + wheel.name: wheel,
            "evidence/archive-checksums.json": archive / "checksums.json",
            "evidence/decision.json": archive / "decision.json",
            "README.md": ROOT / "docs/VISIT_RESIDUAL.md",
            "paper/visit-residual.pdf": ROOT / "paper/visit-residual.pdf",
        }
        for n in ("visit-residual.safetensors", "fit.json", "registration.json"):
            assert source[n] == digest(archive / n)
            files["policy/" + n] = archive / n
        for i, w in selected:
            args = [
                "ask",
                "--checkpoint",
                "release-05",
                "--session-policy",
                "policy",
                "--tool-format",
                "qwen",
                "--trace",
                "--compare-base",
                "--max-turns",
                str(spec["max_turns"]),
                "--max-tokens",
                str(spec["max_tokens"]),
                "--prompt",
                w["questions"][0],
            ]
            for filename in w.get("wheels", []):
                file = ROOT / "data/research_external/wheel-archives/wheels" / filename
                files["source-wheels/" + filename] = file
                args += ["--wheel", "source-wheels/" + filename]
            files["examples/" + w["family"] + ".json"] = (
                json.dumps(args, ensure_ascii=False, indent=2) + "\n"
            ).encode()
        temp_zip = work / "kit.zip"
        hashes = write_archive(temp_zip, files)
        other = work / "independent.zip"
        assert write_archive(other, files) == hashes and digest(other) == digest(
            temp_zip
        )
        installation = work / "installed"
        installation.mkdir()
        with ZipFile(temp_zip) as z:
            z.extractall(installation)
        assert all(digest(installation / n) == v for n, v in hashes.items())
        (installation / "release-05").symlink_to(
            (ROOT / spec["checkpoint"]).resolve(), target_is_directory=True
        )
        site = work / "site"
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
                str(installation / "runtime" / wheel.name),
            ],
            check=True,
        )
        cache = work / "empty-cache"
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        runs = {a: unpack(archive / (a + ".json.gz")) for a in ("candidate", "control")}
        records = []
        for i, w in selected:
            args = read(installation / "examples" / (w["family"] + ".json"))
            actual = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        NETWORK_GUARD + CODE,
                        str(site),
                        *args,
                    ],
                    cwd=installation,
                    env=env,
                )
            )
            assert (
                actual["initial_prompt_identical"]
                and actual["base_source"] == "certified_release_text"
            )
            record = {"id": w["id"], "family": w["family"]}
            for arm, label in (("candidate", "recurrent"), ("control", "base")):
                reference = trace(runs[arm][i]["requests"][0]["trajectory"])
                (measured,) = actual["runs"][label]
                assert trace(measured) == reference
                record[arm] = fingerprint(reference)
            records.append(record)
        assert not any(p.is_file() for p in cache.rglob("*"))
        report = {
            "archive": str(output.relative_to(ROOT)),
            "sha256": digest(temp_zip),
            "bytes": temp_zip.stat().st_size,
            "members": len(hashes),
            "source_revision": revision,
            "wheel": digest(wheel),
            "policy_weights": source["visit-residual.safetensors"],
            "required_release": spec["checkpoint"],
            "archive_manifest": digest(archive / "checksums.json"),
            "cold_installed_pairs": records,
        }
        if receipt.exists():
            assert read(receipt) == report
        if output.exists():
            assert digest(output) == report["sha256"]
        else:
            import shutil

            shutil.copyfile(temp_zip, output)
        save(receipt, report)
        print(report)


if __name__ == "__main__":
    run()
