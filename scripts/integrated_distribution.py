"""Verify offline delivery of the compiled recurrent policy on genuine new and retained requests."""

import json
import os
import shutil
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
from scripts.dialogue_database_sources import CACHE


def run():
    result = ROOT / "research/results/integrated-assistance-01"
    report = ROOT / "docs/measurements/integrated-assistance-distribution.json"
    output = ROOT / "exports/loopqwen-integrated-assistance-01.zip"
    s = read(result / "registration.json")
    scope = read(result / "checksums.json")
    assert (ROOT / "docs/measurements/integrated-assistance-reading.json").is_file()
    revision = read(result / "environment.json")["execution_revision"]
    rows = read(s["material"])["rows"]
    selected = [
        next((i, w) for i, w in enumerate(rows) if w["family"] == g)
        for g in ("database", "open", "dependency")
    ]
    expected = {k: unpack(result / (k + ".json.gz")) for k in ("candidate", "control")}
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        wheel = build(work / "wheel", revision)
        assert digest(wheel) == digest(build(work / "independent", revision))
        files = {
            "runtime/" + wheel.name: wheel,
            "evidence/archive-checksums.json": result / "checksums.json",
            "evidence/decision.json": result / "decision.json",
            "paper/integrated-assistance.pdf": ROOT / "paper/integrated-assistance.pdf",
            "README.md": ROOT / "docs/INTEGRATED_ASSISTANCE.md",
        }
        for name in (
            "registration.json",
            "fit.json",
            "compiled.json",
            "task-merge.json",
            "cross-view.safetensors",
            "tool-adapters.safetensors",
        ):
            assert digest(result / name) == scope[name]
            files["policy/" + name] = result / name
        for _, w in selected:
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
                str(s["max_turns"]),
                "--max-tokens",
                str(s["max_tokens"]),
                "--prompt",
                w["questions"][0],
            ]
            if w["family"] == "database":
                name = w["source"] + ".sqlite"
                path = work / name
                with ZipFile(CACHE) as z:
                    path.write_bytes(z.read(w["sqlite_member"]))
                assert digest(path) == w["sqlite_sha256"]
                files["inputs/" + name] = path
                args += ["--relational-db", "inputs/" + name]
            for name in w.get("wheels", []):
                path = ROOT / "data/research_external/wheel-archives/wheels" / name
                files["inputs/" + name] = path
                args += ["--wheel", "inputs/" + name]
            files["examples/" + w["family"] + ".json"] = (
                json.dumps(args, ensure_ascii=False, indent=2) + "\n"
            ).encode()
        kit = work / "kit.zip"
        identities = write_archive(kit, files)
        copy = work / "again.zip"
        assert write_archive(copy, files) == identities and digest(kit) == digest(copy)
        out = work / "installed"
        out.mkdir()
        with ZipFile(kit) as z:
            assert set(z.namelist()) == set(identities) | {"SHA256SUMS"}
            z.extractall(out)
        assert all(digest(out / name) == sha for name, sha in identities.items())
        (out / "release-05").symlink_to(
            (ROOT / s["checkpoint"]).resolve(), target_is_directory=True
        )
        site = work / "site"
        cache = work / "offline"
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
                str(out / "runtime" / wheel.name),
            ],
            check=True,
        )
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        replay = []
        for i, w in selected:
            args = read(out / "examples" / (w["family"] + ".json"))
            logged = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        NETWORK_GUARD + CODE,
                        str(site),
                        *args,
                    ],
                    cwd=out,
                    env=env,
                )
            )
            assert (
                logged["initial_prompt_identical"]
                and logged["base_source"] == "certified_release_text"
            )
            row = {"id": w["id"], "family": w["family"]}
            for arm, display in (("candidate", "recurrent"), ("control", "base")):
                goal = trace(expected[arm][i]["requests"][0]["trajectory"])
                (measured,) = logged["runs"][display]
                assert trace(measured) == goal
                row[arm] = fingerprint(goal)
            replay.append(row)
        assert not any(p.is_file() for p in cache.rglob("*"))
        record = {
            "archive": str(output.relative_to(ROOT)),
            "bytes": kit.stat().st_size,
            "sha256": digest(kit),
            "members": len(identities),
            "revision": revision,
            "wheel": digest(wheel),
            "packaged_model": scope["cross-view.safetensors"],
            "required_release": s["checkpoint"],
            "evidence_inventory": digest(result / "checksums.json"),
            "cold_installed_pairs": replay,
        }
        if output.is_file():
            assert digest(output) == record["sha256"]
        else:
            shutil.copyfile(kit, output)
        if report.is_file():
            assert read(report) == record
        else:
            save(report, record)
        print(record, flush=True)


if __name__ == "__main__":
    run()
