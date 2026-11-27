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
from loopqwen.research.native_function_study import executions

STUDIES = (
    "observation-review-01",
    "document-frequency-reader-01",
    "question-contrast-01",
    "conversation-document-prefill-01",
    "book-conversation-field-01",
    "report-action-field-01",
)


def full_history(revision, staging):
    # Expose only the requested commit and its ancestors, independent of other local refs.
    bare = staging / "source.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(bare)], check=True)
    objects = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "objects"],
        cwd=ROOT,
        text=True,
    ).strip()
    (bare / "objects/info/alternates").write_text(objects + "\n")
    subprocess.run(
        ["git", "-C", str(bare), "update-ref", "refs/heads/main", revision], check=True
    )
    bundle = staging / "loopqwen-research.bundle"
    subprocess.run(
        [
            "git",
            "-C",
            str(bare),
            "-c",
            "pack.threads=1",
            "bundle",
            "create",
            str(bundle),
            "refs/heads/main",
        ],
        check=True,
    )
    return bundle


def entries(revision, staging):
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
    for dep in wheelhouse(
        ROOT / "exports/field-paper-dependencies-01", only_group="paper"
    ):
        records["paper-tools/" + dep.name] = dep
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
    reporting = read(ROOT / "research/data/report-action-training.json")
    financial = ROOT / "data/research_external/tatqa"
    for split, sha in reporting["sources"].items():
        document = financial / ("tatqa_dataset_" + split + ".json")
        assert digest(document) == sha
        records[document.relative_to(ROOT).as_posix()] = document
    for name, expected in (
        ("LICENSE", "46e27ffbc49c3fd44a9595c2213f8a5f4319a81b53238e49ce1eec39e7e25662"),
        (
            "README.md",
            "85c398173d771a2531abea5b81f3696914296642da2af78c2c34545715c14d5f",
        ),
    ):
        notice = ROOT / "docs/sources/tatqa" / name
        assert digest(notice) == expected
        records[(financial / name).relative_to(ROOT).as_posix()] = notice
    panel = read(ROOT / "research/data/report-action-panel.json")
    for domain in ("report", "distribution"):
        row = next(
            r for r in panel["rows"] if r["family"] == domain and r["expression"]
        )
        records["examples/" + domain + "-question.txt"] = row["questions"][0].encode()
    locked = subprocess.check_output(
        ["git", "show", panel["lock_revision"] + ":uv.lock"], cwd=ROOT
    )
    assert hashlib.sha256(locked).hexdigest() == panel["lock_sha256"]
    records["examples/distribution-inventory.lock"] = locked
    for p in sorted((ROOT / "paper").iterdir()):
        if p.is_file() and p.suffix in {".pdf", ".csv", ".svg", ".typ", ".md", ".json"}:
            records["paper/" + p.name] = p
    for name in (
        "portable-base-wheel-02.json",
        "contrast-reader-installed-02.json",
        "document-reader-installed-02.json",
        "observation-review-base-identity.json",
        "report-distribution-bytes.json",
    ):
        records["verification/" + name] = docs / name
    records["research/data/conversation-document-prefill-tokens.json"] = (
        ROOT / "research/data/conversation-document-prefill-tokens.json"
    )
    records["sources/REVISION.txt"] = (revision + "\n").encode()
    records["sources/loopqwen-research.bundle"] = full_history(revision, staging)
    code = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            revision,
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
    for name, r, (document_revision, _, source, question, _, _) in zip(
        names, panel, documents(), strict=True
    ):
        assert document_revision == r["id"] and question == r["question"]
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
                if name.endswith((".lqzip", ".whl", ".gz", ".pdf", ".bundle"))
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
        receipt = ROOT / "docs/measurements/field-kit-01.json"
        revision = (
            read(receipt)["source_revision"]
            if receipt.exists()
            else subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                revision,
                "--",
                "paper",
                "research",
                "docs/FIELD_KIT.md",
                "docs/sources/tatqa",
            ],
            cwd=ROOT,
            check=True,
        )
        hashes = write_archive(building, entries(revision, Path(staging)))
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
        checkout = tmp / "restored-project"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                "main",
                str(work / "sources/loopqwen-research.bundle"),
                str(checkout),
            ],
            check=True,
        )
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip()
            == revision
        )
        subprocess.run(
            ["git", "fsck", "--full", "--no-reflogs"],
            cwd=checkout,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert digest(checkout / "paper/loopqwen.pdf") == hashes["paper/loopqwen.pdf"]
        registry = read(checkout / "research/registry.json")
        for name, entry in registry["experiments"].items():
            if entry["archive"]:
                assert (
                    digest(checkout / "research/results" / name / "checksums.json")
                    == entry["archive"]
                )
        for name, entry in registry["session_results"].items():
            assert (
                digest(checkout / "research/sessions" / name / "checksums.json")
                == entry["archive"]
            )
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
        subprocess.run(
            [
                python,
                "-I",
                str(checkout / "scripts/paper_induction_evidence.py"),
                "--verify",
            ],
            cwd=checkout,
            check=True,
        )
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
        examples = []
        for name, family in (
            ("question-contrast-01", "ASDiv"),
            ("report-action-field-01", "report"),
            ("report-action-field-01", "distribution"),
        ):
            trial = work / "studies" / name
            spec = read(trial / "registration.json")
            rows = read(work / spec["material"])["rows"]
            candidate = unpack(trial / "candidate.json.gz")
            if family == "ASDiv":
                idx = next(
                    i
                    for i, r in enumerate(candidate)
                    if r["family"] == family and r["requests"][0]["score"]["success"]
                )
                prompt = ["--prompt", rows[idx]["questions"][0]]
            else:
                idx = next(
                    i
                    for i, r in enumerate(rows)
                    if r["family"] == family and r["expression"]
                )
                file = "examples/" + family + "-question.txt"
                assert (work / file).read_text() == rows[idx]["questions"][0]
                prompt = ["--prompt-file", file]
            request = rows[idx]
            args = [
                "ask",
                "--checkpoint",
                "release-05",
                "--tool-adapter",
                "studies/" + name,
                "--tool-format",
                "qwen",
                "--calculator",
                *prompt,
                "--reuse-prefix",
                "--compare-base",
                "--trace",
                "--max-tokens",
                str(spec["max_tokens"]),
                "--max-turns",
                str(spec["max_turns"]),
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
            result = {}
            for side, arm in (
                ("recurrent", candidate),
                ("base", unpack(trial / "control.json.gz")),
            ):
                assert arm[idx]["id"] == request["id"] and len(raw["runs"][side]) == 1
                row = raw["runs"][side][0]
                verified = trace(row)
                assert verified == trace(arm[idx]["requests"][0]["trajectory"])
                result[side] = {
                    "status": row["status"],
                    "answer": row["answer"],
                    "trace_sha256": hashlib.sha256(
                        json.dumps(verified, ensure_ascii=False).encode()
                    ).hexdigest(),
                    "calls": executions(row),
                }
            if family == "ASDiv":
                assert result["recurrent"]["answer"] == request["gold"]
            examples.append(
                {
                    "study": name,
                    "id": request["id"],
                    "domain": family,
                    "command": args,
                    "reference": request["gold"],
                    "models": result,
                }
            )
        assert not any(
            p.suffix in {".bin", ".safetensors"} for p in (tmp / "empty").rglob("*")
        )
        paper_wheels = sorted((work / "paper-tools").glob("*.whl"))
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
                *map(str, paper_wheels),
            ],
            cwd=work,
            check=True,
        )
        rebuilt = tmp / "reconstructed.pdf"
        pages = int(
            subprocess.check_output(
                [
                    python,
                    "-I",
                    "-c",
                    NETWORK_GUARD
                    + "import sys,typst;from pypdf import PdfReader;typst.compile(sys.argv[1],output=sys.argv[2],timestamp=0);print(len(PdfReader(sys.argv[2]).pages))",
                    str(work / "paper/loopqwen.typ"),
                    str(rebuilt),
                ],
                cwd=work,
                env=env,
                text=True,
            )
        )
        assert digest(rebuilt) == hashes["paper/loopqwen.pdf"]
        record = {
            "name": target.name,
            "bytes": building.stat().st_size,
            "sha256": digest(building),
            "verified_members": len(hashes),
            "restored_model_files": len(original),
            "model_sha256": hashes["model/tool-language-packed-05.lqzip"],
            "wheel_sha256": digest(wheel),
            "pdf_sha256": hashes["paper/loopqwen.pdf"],
            "rebuilt_pdf_pages": pages,
            "locked_paper_wheels": len(paper_wheels),
            "source_revision": revision,
            "verified_git_history": True,
            "recounted_matched_induction_items": 128,
            "registrations": len(registry["experiments"]),
            "study_archives": sum(
                bool(e["archive"]) for e in registry["experiments"].values()
            )
            + len(registry["session_results"]),
            "empty_cache": True,
            "outside_checkout": True,
            "fresh_venv": True,
            "locked_dependency_wheels": len(
                list((work / "dependencies").glob("*.whl"))
            ),
            "network_blocked": True,
            "initial_prompt_identical": True,
            "verified_examples": examples,
        }
        if receipt.exists():
            assert read(receipt) == record
        if not target.exists():
            os.replace(building, target)
        if not receipt.exists():
            save(receipt, record)
        print(target, record["bytes"], record["sha256"])


if __name__ == "__main__":
    main()
