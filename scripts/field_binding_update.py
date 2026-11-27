"""Build the small field-kit update and replay both installed decoders without a network."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from field_kit import write_archive
from field_update_context import prepare as context_examples
from field_update_context import verify as verify_context
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

STUDY = "instruction-interpolation-01"
BINDING = "role-binding-01"
WHEEL_REV = "1f1503f"
ARCHIVE = ROOT / "exports/loopqwen-field-update-02.zip"
RECEIPT = ROOT / "docs/measurements/field-binding-update-02.json"


def history_update(folder, revision, parent):
    """Carry the exact new source, evidence and commit sequence on top of kit 01."""
    bare = folder / "source.git"
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
    bundle = folder / "field-update.bundle"
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
            "^" + parent,
        ],
        check=True,
    )
    return bundle


def main():
    trial = ROOT / "research/results" / STUDY
    spec = read(trial / "registration.json")
    panel = read(ROOT / spec["material"])
    kit = read(ROOT / "docs/measurements/field-kit-01.json")
    revision = (
        read(RECEIPT)["source_revision"]
        if RECEIPT.is_file()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    original = ROOT / "exports" / kit["name"]
    assert digest(original) == kit["sha256"]
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        archive = tmp / ARCHIVE.name
        work = tmp / "replay"
        wheel = build(tmp / "build", WHEEL_REV)
        assert digest(build(tmp / "repeat", WHEEL_REV)) == digest(wheel)
        files = {
            "README.md": ROOT / "docs/INSTRUCTION_UPDATE.md",
            "runtime/" + wheel.name: wheel,
            spec["material"]: ROOT / spec["material"],
            "diagnostic/activation.json": ROOT
            / "docs/measurements/instruction-interpolation-activation.json",
            "sources/WSCollection.xml": ROOT
            / "data/research_external/winograd/WSCollection.xml",
            "sources/uv.lock": subprocess.check_output(
                ["git", "show", panel["lock_revision"] + ":uv.lock"], cwd=ROOT
            ),
        }
        assert digest(files["sources/WSCollection.xml"]) == panel["source_sha256"]
        for name, body in panel["documents"].items():
            files["inventories/" + name] = body.encode()
        for path in trial.iterdir():
            files["studies/" + STUDY + "/" + path.name] = path
        for path in sorted((ROOT / "paper").glob("instruction-interpolation*")):
            files["paper/" + path.name] = path
        for name in (
            "scripts/interpolation_activation.py",
            "scripts/interpolation_material.py",
            "scripts/paper_instruction_interpolation.py",
            "research/diagnostics/instruction-interpolation-activation-01.json",
        ):
            files[name] = ROOT / name
        files["sources/field-update.bundle"] = history_update(
            tmp, revision, kit["source_revision"]
        )
        files["sources/REVISION.txt"] = (revision + "\n").encode()
        binding = ROOT / "research/results" / BINDING
        binding_spec = read(binding / "registration.json")
        fresh = read(ROOT / binding_spec["material"])
        files.update(
            {
                name: ROOT / name
                for name in (
                    binding_spec["material"],
                    binding_spec["training"],
                    "scripts/binding_material.py",
                    "scripts/paper_role_binding.py",
                    "scripts/field_binding_update.py",
                    "scripts/report_distribution_bytes.py",
                    "scripts/binding_source_check.py",
                    "scripts/field_update_context.py",
                    "scripts/role_binding_reading.py",
                    "scripts/role_binding_exposure.py",
                    "docs/measurements/role-binding-training-exposure.json",
                    "docs/measurements/role-binding-activation.json",
                    "research/diagnostics/role-binding-activation-01.json",
                    "research/diagnostics/role-binding-reading-audit.json",
                    "docs/measurements/role-binding-reading-decisions.json",
                    "scripts/assemble_field_paper.py",
                    "docs/measurements/role-binding-source-check.json",
                    "docs/measurements/keyed-inventory-bytes.json",
                    "docs/sources/wikisql/README.md",
                    "docs/sources/wikisql/LICENSE",
                    "research/experiments/role-binding-01.json",
                    "research/experiments/instruction-interpolation-01.json",
                )
            }
        )
        for path in binding.iterdir():
            files["studies/" + BINDING + "/" + path.name] = path
        for pattern in ("role-binding*", "loopqwen-complete.pdf"):
            for path in sorted((ROOT / "paper").glob(pattern)):
                files["paper/" + path.name] = path
        for name, body in fresh["documents"].items():
            files["inventories/" + name] = body.encode()
        for short, rel, key in (
            ("winogrande-1.1.zip", "winogrande/source.zip", "winogrande"),
            ("wikisql.tar.bz2", "wikisql/source.tar.bz2", "wikisql"),
        ):
            path = ROOT / "data/research_external" / rel
            assert digest(path) == fresh["sources"][key]["sha256"]
            files["sources/" + short] = path
        examples = []
        studies = (
            (STUDY, spec, panel, ("reference", "inventory")),
            (BINDING, binding_spec, fresh, ("meaning", "table")),
        )
        for name, policy, data, families in studies:
            for family in families:
                i = next(i for i, r in enumerate(data["rows"]) if r["family"] == family)
                row = data["rows"][i]
                args = [
                    "ask",
                    "--checkpoint",
                    "release-05",
                    "--session-policy",
                    "studies/" + name,
                    "--tool-format",
                    "qwen",
                    "--compare-base",
                    "--trace",
                    "--reuse-prefix",
                    "--max-tokens",
                    str(policy["max_tokens"]),
                    "--max-turns",
                    str(policy["max_turns"]),
                ]
                if "document" in row:
                    args.extend(["--keyed-csv", "inventories/" + row["document"]])
                if family == "inventory":
                    args.append("--calculator")
                for j, q in enumerate(row["questions"]):
                    args.extend(["--follow-up" if j else "--prompt", q])
                files["examples/" + family + ".json"] = (
                    json.dumps(args, ensure_ascii=False, indent=2) + "\n"
                ).encode()
                examples.append((name, family, i, row["id"]))
        long_examples = context_examples(ROOT, files)
        hashes = write_archive(archive, files)
        with ZipFile(archive) as z:
            assert set(z.namelist()) == set(hashes) | {"SHA256SUMS"}
            z.extractall(work)
        assert all(
            digest(work / k) == v for k, v in hashes.items()
        ) and not work.resolve().is_relative_to(ROOT)
        with ZipFile(original) as z:
            for name in z.namelist():
                if name.startswith(
                    ("model/", "dependencies/", "paper-tools/")
                ) or name in ("paper/loopqwen.pdf", "sources/loopqwen-research.bundle"):
                    z.extract(name, work)
        checkout = tmp / "restored-source.git"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--bare",
                str(work / "sources/loopqwen-research.bundle"),
                str(checkout),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "bundle",
                "verify",
                str(work / "sources/field-update.bundle"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "--quiet",
                str(work / "sources/field-update.bundle"),
                "refs/heads/main:refs/heads/main",
            ],
            check=True,
        )
        assert (
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "refs/heads/main"], text=True
            ).strip()
            == revision
        )
        for path in (
            "paper/loopqwen-complete.pdf",
            "research/results/" + BINDING + "/tool-adapters.safetensors",
        ):
            restored = subprocess.check_output(
                ["git", "-C", str(checkout), "show", revision + ":" + path]
            )
            assert restored == (ROOT / path).read_bytes()
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
                str(work / "runtime" / wheel.name),
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(["uv", "pip", "check", "--python", python], cwd=work, check=True)
        site = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
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
            TRANSFORMERS_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        subprocess.run(
            [
                python,
                "-I",
                "-c",
                NETWORK_GUARD
                + "import runpy;runpy.run_module('loopqwen.weight_transport',run_name='__main__')",
                "unpack",
                "model/tool-language-packed-05.lqzip",
                "release-05",
            ],
            cwd=work,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert all(
            digest(work / "release-05" / p) == v
            for p, v in read(
                ROOT / "research/results/packed-release-retention-04/export-files.json"
            ).items()
        )
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
                *map(str, sorted((work / "paper-tools").glob("*.whl"))),
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [python, "-I", "scripts/assemble_field_paper.py", "paper", "--verify"],
            cwd=work,
            env=env,
            check=True,
        )

        def execute(args):
            return json.loads(
                subprocess.check_output(
                    [python, "-I", "-c", NETWORK_GUARD + CODE, site, *args],
                    cwd=work,
                    env=env,
                )
            )

        replays = []
        for study, family, i, ident in examples:
            args = read(work / "examples" / (family + ".json"))
            actual = execute(args)
            assert (
                actual["initial_prompt_identical"]
                and actual["base_source"] == "certified_release_text"
            )
            hashes_by_arm = {}
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                registered = unpack(work / "studies" / study / (arm + ".json.gz"))[i]
                traces = [trace(t) for t in actual["runs"][side]]
                assert traces == [
                    trace(r["trajectory"]) for r in registered["requests"]
                ]
                hashes_by_arm[side] = fingerprint(traces)
            replays.append(
                {
                    "family": family,
                    "id": ident,
                    "study": study,
                    "requests_per_arm": len(actual["runs"]["base"]),
                    "trace_sha256": hashes_by_arm,
                }
            )
        long_results = verify_context(
            work,
            long_examples,
            execute,
            ROOT / "docs/measurements/field-binding-long-replay.json",
        )
        assert not any(p.is_file() for p in cache.rglob("*"))
        record = {
            "archive": ARCHIVE.name,
            "source_revision": revision,
            "incremental_history_verified": True,
            "sha256": digest(archive),
            "bytes": archive.stat().st_size,
            "verified_payloads": len(hashes),
            "qualified_context_inputs": [
                r["actual_initial_tokens"] for r in long_results
            ],
            "rebuilt_pdf_sha256": digest(work / "paper/loopqwen-complete.pdf"),
            "required_kit_sha256": kit["sha256"],
            "wheel_sha256": digest(wheel),
            "wheel_source_revision": subprocess.check_output(
                ["git", "rev-parse", WHEEL_REV], text=True
            ).strip(),
            "empty_cache": True,
            "fresh_venv": True,
            "network_blocked": True,
            "examples": replays,
        }
        if ARCHIVE.exists():
            assert digest(ARCHIVE) == record["sha256"]
        else:
            ARCHIVE.write_bytes(archive.read_bytes())
        if RECEIPT.exists():
            assert read(RECEIPT) == record
        else:
            save(RECEIPT, record)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
