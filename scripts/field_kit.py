"""Assemble and verify the qualified model, installed runtime, paper and recent field evidence."""

import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from document_frequency_material import documents
from field_wheelhouse import wheelhouse
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save

SOURCE_REVISION = "22d49272762904e5c38a4b9e22587651172e0a27"

STUDIES = (
    "observation-review-01",
    "document-frequency-reader-01",
    "question-contrast-01",
    "conversation-document-prefill-01",
    "book-conversation-field-01",
)


def entries():
    docs = ROOT / "docs/measurements"
    wheel = read(docs / "portable-base-wheel-02.json")
    runtime = ROOT / "exports/portable-base-runtime-02" / wheel["wheel"]
    weights = ROOT / "exports/tool-language-packed-05.lqzip"
    assert digest(runtime) == wheel["sha256"]
    assert (
        digest(weights)
        == read(ROOT / "research/results/transport-lossless-04/report.json")["sha256"]
    )
    records = {
        "README.md": ROOT / "docs/FIELD_KIT.md",
        "LICENSE": ROOT / "LICENSE",
        "runtime/" + runtime.name: runtime,
        "model/" + weights.name: weights,
    }
    for dep in wheelhouse(ROOT / "exports/field-dependencies-01"):
        records["dependencies/" + dep.name] = dep
    registry = read(ROOT / "research/registry.json")
    for name in STUDIES:
        folder = ROOT / "research/results" / name
        assert (
            digest(folder / "checksums.json")
            == registry["experiments"][name]["archive"]
        )
        index = read(folder / "checksums.json")
        protocol = read(folder / "registration.json")
        for field in ("material", "training"):
            relative = protocol.get(field, "")
            if isinstance(relative, str) and relative.startswith("research/data/"):
                assert (
                    digest(ROOT / relative) == registry["datasets"][relative]["sha256"]
                )
                records[relative] = ROOT / relative
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                relative = p.relative_to(folder).as_posix()
                assert relative == "checksums.json" or digest(p) == index[relative]
                records["studies/" + name + "/" + relative] = p
    for p in sorted((ROOT / "paper").iterdir()):
        if p.is_file() and p.suffix in {".pdf", ".csv", ".svg", ".typ", ".md", ".json"}:
            records["paper/" + p.name] = p
    for name in (
        "portable-base-wheel-02.json",
        "contrast-reader-installed-02.json",
        "document-reader-installed-02.json",
        "observation-review-base-identity.json",
    ):
        records["verification/" + name] = docs / name
    records["research/data/conversation-document-prefill-tokens.json"] = (
        ROOT / "research/data/conversation-document-prefill-tokens.json"
    )
    records["sources/REVISION.txt"] = (SOURCE_REVISION + "\n").encode()
    code = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            SOURCE_REVISION,
            "loopqwen",
            "scripts",
            "tests",
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "AGENTS.md",
            "RESEARCH_GOAL.md",
        ],
        cwd=ROOT,
    )
    records["sources/loopqwen-source.tar.gz"] = gzip.compress(code, mtime=0)
    panel = read(ROOT / "research/data/document-frequency-panel.json")["panel"]
    names = [
        "complete-tree.txt",
        "complete-tree-next-revision.txt",
        "complete-lockfile.txt",
    ]
    for name, r, (revision, _, source, question, _, _) in zip(
        names, panel, documents(), strict=True
    ):
        assert revision == r["id"] and question == r["question"]
        assert hashlib.sha256(source.encode()).hexdigest() == r["source_sha256"]
        records["examples/" + name] = (source + "\n\n" + question).encode()
    for book in read(ROOT / "research/data/book-conversation-panel.json")["items"]:
        file = ROOT / book["file"]
        assert digest(file) == book["sha256"] and file.stat().st_size == book["size"]
        records[book["file"]] = file
    return records


def write_archive(path, files):
    hashes = {}
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:

        def put(name, src):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = (
                zipfile.ZIP_STORED
                if name.endswith((".lqzip", ".whl", ".gz", ".pdf"))
                else zipfile.ZIP_DEFLATED
            )
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            h = hashlib.sha256()
            with archive.open(entry, "w", force_zip64=True) as out:
                if isinstance(src, Path):
                    with src.open("rb") as inp:
                        while block := inp.read(1 << 20):
                            h.update(block)
                            out.write(block)
                else:
                    out.write(src)
                    h.update(src)
            return h.hexdigest()

        for name, src in sorted(files.items()):
            hashes[name] = put(name, src)
        put(
            "SHA256SUMS",
            "".join(f"{h}  {name}\n" for name, h in hashes.items()).encode(),
        )
    return hashes


def main():
    directory = ROOT / "exports"
    target = directory / "loopqwen-field-kit-01.zip"
    with (
        tempfile.TemporaryDirectory(dir=directory) as staging,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp = Path(tmp)
        building = Path(staging) / target.name
        hashes = write_archive(building, entries())
        if target.exists():
            assert digest(target) == digest(building)
        work = tmp / "field"
        with zipfile.ZipFile(building) as archive:
            assert set(archive.namelist()) == set(hashes) | {"SHA256SUMS"}
            archive.extractall(work)
        assert (work / "SHA256SUMS").read_text() == "".join(
            f"{h}  {name}\n" for name, h in hashes.items()
        )
        assert all(digest(work / k) == v for k, v in hashes.items())
        assert not work.resolve().is_relative_to(ROOT)
        venv = tmp / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--offline",
                "--no-project",
                "--python",
                sys.executable,
                str(venv),
            ],
            check=True,
        )
        python = str(venv / "bin/python")
        wheel = next((work / "runtime").glob("*.whl"))
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
                str(work / "dependencies"),
                str(wheel),
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(["uv", "pip", "check", "--python", python], cwd=work, check=True)
        site = Path(
            subprocess.check_output(
                [
                    python,
                    "-I",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                cwd=work,
                text=True,
            ).strip()
        )
        assert site.is_relative_to(venv)
        env = {
            **os.environ,
            "HF_HOME": str(tmp / "empty"),
            "HF_HUB_CACHE": str(tmp / "empty/hub"),
            "HUGGINGFACE_HUB_CACHE": str(tmp / "empty/hub"),
            "TRANSFORMERS_CACHE": str(tmp / "empty/hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('loopqwen.weight_transport',run_name='__main__')"
        subprocess.run(
            [
                python,
                "-I",
                "-c",
                NETWORK_GUARD + bootstrap,
                str(site),
                "unpack",
                "model/tool-language-packed-05.lqzip",
                "release-05",
            ],
            cwd=work,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        restored = work / "release-05"
        original = read(
            ROOT / "research/results/packed-release-retention-04/export-files.json"
        )
        assert all(digest(restored / p) == h for p, h in original.items())
        trial = work / "studies/question-contrast-01"
        candidate = unpack(trial / "candidate.json.gz")
        control = unpack(trial / "control.json.gz")
        idx = next(
            i
            for i, r in enumerate(candidate)
            if r["family"] == "ASDiv" and r["requests"][0]["score"]["success"]
        )
        original_request = read(work / read(trial / "registration.json")["material"])[
            "rows"
        ][idx]
        assert original_request["id"] == candidate[idx]["id"] == control[idx]["id"]
        args = [
            "ask",
            "--checkpoint",
            "release-05",
            "--tool-adapter",
            "studies/question-contrast-01",
            "--tool-format",
            "qwen",
            "--calculator",
            "--prompt",
            original_request["questions"][0],
            "--reuse-prefix",
            "--compare-base",
            "--trace",
            "--max-tokens",
            "384",
            "--max-turns",
            "3",
        ]
        raw = json.loads(
            subprocess.check_output(
                [python, "-I", "-c", NETWORK_GUARD + CODE, str(site), *args],
                cwd=work,
                env=env,
            )
        )
        assert (
            raw["base_source"] == "certified_release_text"
            and raw["initial_prompt_identical"]
        )
        assert all(len(v) == 1 for v in raw["runs"].values())
        for side, arm in (("recurrent", candidate), ("base", control)):
            assert trace(raw["runs"][side][0]) == trace(
                arm[idx]["requests"][0]["trajectory"]
            )
        assert raw["runs"]["recurrent"][0]["answer"] == original_request["gold"]
        assert not any(
            p.suffix in {".bin", ".safetensors"} for p in (tmp / "empty").rglob("*")
        )
        # Strip the temporary root from the receipt. Arguments inside the kit are all relative.
        answer = {
            s: {
                "answer": rows[0]["answer"],
                "status": rows[0]["status"],
                "trace_sha256": hashlib.sha256(
                    json.dumps(trace(rows[0]), ensure_ascii=False).encode()
                ).hexdigest(),
                "calls": [
                    {k: t[k] for k in ("call", "observation")}
                    for t in rows[0]["turns"]
                    if "call" in t and "observation" in t
                ],
            }
            for s, rows in raw["runs"].items()
        }
        record = {
            "name": target.name,
            "bytes": building.stat().st_size,
            "sha256": digest(building),
            "verified_members": len(hashes),
            "restored_model_files": len(original),
            "model_sha256": hashes["model/tool-language-packed-05.lqzip"],
            "wheel_sha256": digest(wheel),
            "pdf_sha256": hashes["paper/loopqwen.pdf"],
            "source_revision": SOURCE_REVISION,
            "empty_cache": True,
            "outside_checkout": True,
            "fresh_venv": True,
            "locked_dependency_wheels": len(
                list((work / "dependencies").glob("*.whl"))
            ),
            "network_blocked": True,
            "initial_prompt_identical": True,
            "command": args,
            "replayed_registered_id": original_request["id"],
            "example": answer,
        }
        receipt = ROOT / "docs/measurements/field-kit-01.json"
        if receipt.exists():
            assert read(receipt) == record
        if not target.exists():
            os.replace(building, target)
        if not receipt.exists():
            save(receipt, record)
        print(target, record["bytes"], record["sha256"])


if __name__ == "__main__":
    main()
