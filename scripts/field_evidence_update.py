"""Build and independently replay the complete changed-file research update outside the checkout."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from field_binding_update import history_update
from field_kit import write_archive
from field_update_context import prepare
from field_update_context import verify as verify_context
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

NAME = "file-evidence-accumulated-01"
WHEEL = "b980402"
DEST = ROOT / "exports/loopqwen-field-update-03.zip"
RECEIPT = ROOT / "docs/measurements/field-evidence-update-03.json"


def cmd(*args, **kwargs):
    subprocess.run(list(args), check=True, **kwargs)


def inventory(tmp, revision, wheel, kit):
    spec = read(ROOT / "research/experiments" / (NAME + ".json"))
    files = {
        "README.md": ROOT / "docs/FILE_EVIDENCE_UPDATE.md",
        "runtime/" + wheel.name: wheel,
        "sources/field-update.bundle": history_update(
            tmp, revision, kit["source_revision"]
        ),
    }
    for study in (NAME, "file-evidence-01", "revision-evidence-01"):
        folder = ROOT / "research/results" / study
        for path in sorted(folder.iterdir()):
            files["studies/" + study + "/" + path.name] = path
    for name in ("adapters.safetensors", "state.json"):
        files["compact-source/" + name] = (
            ROOT / "research/results/entailment-triplets-01/model" / name
        )
    for name in ("train", "test"):
        path = ROOT / "data/research_external/vitaminc" / (name + ".jsonl")
        assert digest(path) == spec["sources"][name]
        files["sources/vitaminc/" + path.name] = path
    for name in ("VITAMINC-LICENSE.txt", "WIKISQL-BSD.txt"):
        files["sources/" + name] = ROOT / "research/licenses" / name
    for path in sorted((ROOT / "paper").iterdir()):
        if path.name in (
            "loopqwen-complete.pdf",
            "loopqwen-research.pdf",
            "loopqwen-report.pdf",
        ) or path.name.startswith(
            (
                "file-evidence",
                "revision-evidence",
                "role-binding",
                "instruction-interpolation",
            )
        ):
            files["paper/" + path.name] = path
    for name in ("assemble_field_paper.py", "assemble_evidence_paper.py"):
        files["scripts/" + name] = ROOT / "scripts" / name
    for path in sorted((ROOT / "research/diagnostics").glob("*")):
        if path.name.startswith(("file-evidence", "revision-")):
            files["sources/diagnostics/" + path.name] = path
    for path in sorted((ROOT / "docs/measurements").glob("*")):
        if (
            path.name.startswith(("file-evidence-", "revision-"))
            and path.name not in (RECEIPT.name, "file-evidence-long-replay.json")
        ) or path.name == "compact-revision-runtime.json":
            files["verification/" + path.name] = path
    for path in [
        ROOT / spec["material"],
        ROOT / spec["training"],
        ROOT / "research/data/revision-evidence-01.json",
        ROOT / "research/data/role-binding-training.json",
    ]:
        files["material/" + path.name] = path
    rows = read(ROOT / spec["material"])["rows"]
    examples = []
    for family in ("file", "inline"):
        group = next(w["cluster"] for w in rows if w["family"] == family)
        for j, (i, w) in enumerate(
            (i, w) for i, w in enumerate(rows) if w["cluster"] == group
        ):
            key = f"{family}-{j}"
            args = [
                "ask",
                "--checkpoint",
                "release-05",
                "--session-policy",
                "studies/" + NAME,
                "--tool-format",
                "qwen",
                "--compare-base",
                "--trace",
                "--max-tokens",
                str(spec["max_tokens"]),
                "--max-turns",
                str(spec["max_turns"]),
                "--prompt",
                w["questions"][0],
            ]
            if family == "file":
                path = "examples/" + key + "/clippings.csv"
                files[path] = w["file"].encode()
                args.extend(["--keyed-csv", path])
            files["examples/" + key + ".json"] = (
                json.dumps(args, indent=2) + "\n"
            ).encode()
            examples.append((key, i, w["id"]))
    long = prepare(ROOT, files)
    # Snapshot tracked inputs from the precise source commit on later rebuilds.
    tracked = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True
        ).splitlines()
    )
    for k, path in list(files.items()):
        if (
            isinstance(path, Path)
            and path.is_relative_to(ROOT)
            and (rel := path.relative_to(ROOT).as_posix()) in tracked
        ):
            files[k] = subprocess.check_output(
                ["git", "show", revision + ":" + rel], cwd=ROOT
            )
    return files, examples, long


def main():
    kit = read(ROOT / "docs/measurements/field-kit-01.json")
    parent = ROOT / "exports" / kit["name"]
    assert digest(parent) == kit["sha256"]
    revision = (
        read(RECEIPT)["source_revision"]
        if RECEIPT.is_file()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        work = tmp / "installed"
        work.mkdir()
        wheel = build(tmp / "wheels", WHEEL)
        assert digest(build(tmp / "repeat", WHEEL)) == digest(wheel)
        files, examples, long = inventory(tmp, revision, wheel, kit)
        archive = tmp / DEST.name
        hashes = write_archive(archive, files)
        with ZipFile(archive) as z:
            assert set(z.namelist()) == set(hashes) | {"SHA256SUMS"}
            z.extractall(work)
        assert all(
            digest(work / k) == v for k, v in hashes.items()
        ) and not work.is_relative_to(ROOT)
        with ZipFile(parent) as z:
            for name in z.namelist():
                if name.startswith(
                    ("dependencies/", "model/", "paper-tools/")
                ) or name in ("sources/loopqwen-research.bundle", "paper/loopqwen.pdf"):
                    z.extract(name, work)
        bare = tmp / "restored.git"
        cmd(
            "git",
            "clone",
            "--quiet",
            "--bare",
            work / "sources/loopqwen-research.bundle",
            bare,
        )
        cmd(
            "git",
            "-C",
            bare,
            "fetch",
            "--quiet",
            work / "sources/field-update.bundle",
            "refs/heads/main:refs/heads/main",
        )
        assert (
            subprocess.check_output(
                ["git", "-C", bare, "rev-parse", "main"], text=True
            ).strip()
            == revision
        )
        for source, target in (
            ("paper/loopqwen-report.pdf", "paper/loopqwen-report.pdf"),
            (
                "research/results/" + NAME + "/tool-adapters.safetensors",
                "studies/" + NAME + "/tool-adapters.safetensors",
            ),
        ):
            assert (
                subprocess.check_output(
                    ["git", "-C", bare, "show", revision + ":" + source]
                )
                == (work / target).read_bytes()
            )
        venv = tmp / "venv"
        cmd("uv", "venv", "--no-project", "--offline", "--python", sys.executable, venv)
        python = str(venv / "bin/python")
        cmd(
            "uv",
            "--no-cache",
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--python",
            python,
            "--find-links",
            work / "dependencies",
            work / "runtime" / wheel.name,
            *sorted((work / "paper-tools").glob("*.whl")),
        )
        cmd("uv", "pip", "check", "--python", python)
        site = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
        cache = tmp / "empty"
        cache.mkdir()
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_OFFLINE="1",
        )

        def module(mod, args):
            code = (
                "import runpy;runpy.run_module(" + repr(mod) + ",run_name='__main__')"
            )
            cmd(
                python,
                "-I",
                "-c",
                NETWORK_GUARD + code,
                *args,
                env=env,
                cwd=work,
                stdout=subprocess.DEVNULL,
            )

        module(
            "loopqwen.weight_transport",
            ["unpack", "model/tool-language-packed-05.lqzip", "release-05"],
        )
        for script in (
            "scripts/assemble_field_paper.py",
            "scripts/assemble_evidence_paper.py",
        ):
            cmd(python, "-I", script, "paper", "--verify", env=env, cwd=work)

        def execute(args):
            return json.loads(
                subprocess.check_output(
                    [python, "-I", "-c", NETWORK_GUARD + CODE, site, *args],
                    cwd=work,
                    env=env,
                )
            )

        short = []
        for key, i, ident in examples:
            actual = execute(read(work / "examples" / (key + ".json")))
            assert (
                actual["initial_prompt_identical"]
                and actual["base_source"] == "certified_release_text"
            )
            got = {}
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                r = unpack(work / "studies" / NAME / (arm + ".json.gz"))[i]
                reference = [trace(q["trajectory"]) for q in r["requests"]]
                current = [trace(q) for q in actual["runs"][side]]
                assert r["id"] == ident and current == reference
                got[side] = fingerprint(current)
            short.append({"example": key, "id": ident, "traces": got})
        module(
            "loopqwen.compact_entailment",
            ["release-05", "compact-source", "compact-reader"],
        )
        from_model = read(work / "studies/revision-evidence-01/compact-entailment.json")
        assert (
            digest(work / "compact-reader/model.safetensors")
            == from_model["files"]["model.safetensors"]
        )
        panel = read(work / "material/revision-evidence-01.json")
        row = panel[0]
        actual = execute(
            [
                "generate",
                "--checkpoint",
                "compact-reader",
                "--compare-base",
                "--base-checkpoint",
                "release-05",
                "--thinking",
                "--trace",
                "--max-tokens",
                "768",
                "--prompt",
                row["body"],
            ]
        )
        assert (
            actual["initial_prompt_identical"]
            and actual["base_source"] == "certified_release_text"
        )
        for side, arm in (("recurrent", "candidate"), ("base", "base")):
            r = unpack(work / "studies/revision-evidence-01" / (arm + ".json.gz"))[0]
            out = actual["runs"][side]
            assert out["initial_sha256"] == fingerprint(row["prompt_ids"]) and all(
                r[k] == out[k] for k in ("token_ids", "kv_history", "text")
            )
        long_records = verify_context(
            work,
            long,
            execute,
            ROOT / "docs/measurements/file-evidence-long-replay.json",
        )
        assert not any(p.is_file() for p in cache.rglob("*"))
        receipt = {
            "archive": DEST.name,
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
            "verified_payloads": len(hashes),
            "source_revision": revision,
            "wheel_source": subprocess.check_output(
                ["git", "rev-parse", WHEEL], cwd=ROOT, text=True
            ).strip(),
            "wheel_sha256": digest(wheel),
            "required_kit_sha256": kit["sha256"],
            "outside_checkout": True,
            "empty_model_cache": True,
            "network_blocked": True,
            "history_restored": True,
            "examples": short,
            "compact_weight_sha256": from_model["files"]["model.safetensors"],
            "compact_replay": row["id"],
            "default_context_lengths": [
                r["actual_initial_tokens"] for r in long_records
            ],
            "paper_sha256": digest(work / "paper/loopqwen-report.pdf"),
        }
        if DEST.exists():
            assert digest(DEST) == receipt["sha256"]
        else:
            stage = DEST.with_suffix(".tmp")
            shutil.copyfile(archive, stage)
            stage.replace(DEST)
        if RECEIPT.exists():
            assert read(RECEIPT) == receipt
        else:
            save(RECEIPT, receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
