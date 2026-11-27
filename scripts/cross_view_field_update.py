"""Package the trained student and verify five real requests from a fresh installation."""

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
from field_wheelhouse import wheelhouse
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, build, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

NAME = "cross-view-reading-01"
WHEEL = "7f9ab69"
RECEIPT = ROOT / "docs/measurements/cross-view-field-update-04.json"
ZIP = ROOT / "exports/loopqwen-field-update-04.zip"


def command(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def contents(tmp, revision, kit):
    source = ROOT / "research/results" / NAME
    spec = read(source / "registration.json")
    panel = read(ROOT / spec["material"])
    wheel = build(tmp / "wheel", WHEEL)
    assert digest(wheel) == digest(build(tmp / "duplicate", WHEEL))
    files = {
        "runtime/" + wheel.name: wheel,
        "README.md": ROOT / "docs/CROSS_VIEW_UPDATE.md",
        "sources/update.bundle": history_update(tmp, revision, kit["source_revision"]),
    }
    with ZipFile(ROOT / "exports" / kit["name"]) as parent:
        available = set(parent.namelist())
    for dep in wheelhouse(
        ROOT / "exports/field-figure-dependencies-04", only_group="figures"
    ):
        if "dependencies/" + dep.name not in available:
            files["figure-tools/" + dep.name] = dep
    for path in source.iterdir():
        files["policy/" + path.name] = path
    for p in (ROOT / "paper").iterdir():
        if p.name.startswith("cross-view") or p.name in (
            "system-comparisons.csv",
            "loopqwen-report.pdf",
            "loopqwen-study.pdf",
        ):
            files["paper/" + p.name] = p
    files["scripts/assemble_evidence_paper.py"] = (
        ROOT / "scripts/assemble_evidence_paper.py"
    )
    for p in (ROOT / "docs/measurements").glob("cross-view-*"):
        if p != RECEIPT:
            files["verification/" + p.name] = p
    for p in (ROOT / "research/diagnostics").glob("cross-view-*"):
        files["sources/" + p.name] = p
    for name in ("QUOREF-SOURCES.md", "WIKISQL-BSD.txt"):
        files["sources/" + name] = ROOT / "research/licenses" / name
    for path in (ROOT / spec["material"], ROOT / spec["training"]):
        files["material/" + path.name] = path
    for label, rel in (
        ("squad", "squad-train-v1.1.json"),
        ("quoref", "quoref/train-dev.zip"),
    ):
        path = ROOT / "data/research_external" / rel
        assert digest(path) == spec["source_hashes"][label]
        files["sources/" + rel] = path
    article = panel["articles"][0]["page"]
    chosen = [(i, w) for i, w in enumerate(panel["rows"]) if w["cluster"] == article]
    chosen.extend(
        [
            next(
                (i, w)
                for i, w in enumerate(panel["rows"])
                if w["family"] == "retention"
            )
        ]
    )
    assert len(chosen) == 5
    cases = []
    for j, (i, w) in enumerate(chosen):
        key = f"case-{j}-{w['family']}"
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
        if w["document"]:
            path = "examples/" + w["document"]
            files[path] = panel["documents"][w["document"]].encode()
            args.extend(["--keyed-csv", path])
        files["examples/" + key + ".json"] = (
            json.dumps(args, indent=2) + "\n"
        ).encode()
        cases.append((key, i, w["id"]))
    tracked = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True
        ).splitlines()
    )
    for k, path in list(files.items()):
        if isinstance(path, Path) and path.is_relative_to(ROOT):
            rel = path.relative_to(ROOT).as_posix()
            if rel in tracked:
                files[k] = subprocess.check_output(
                    ["git", "show", revision + ":" + rel], cwd=ROOT
                )
    return files, cases, wheel.name


def main():
    kit = read(ROOT / "docs/measurements/field-kit-01.json")
    parent = ROOT / "exports" / kit["name"]
    assert digest(parent) == kit["sha256"]
    revision = (
        read(RECEIPT)["source_revision"]
        if RECEIPT.exists()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    with TemporaryDirectory() as d:
        tmp = Path(d)
        work = tmp / "installed"
        work.mkdir()
        assert not work.is_relative_to(ROOT)
        files, cases, wheel = contents(tmp, revision, kit)
        archive = tmp / ZIP.name
        hashes = write_archive(archive, files)
        with ZipFile(archive) as z:
            assert set(z.namelist()) == set(hashes) | {"SHA256SUMS"}
            z.extractall(work)
        for k, v in hashes.items():
            assert digest(work / k) == v
        with ZipFile(parent) as z:
            for k in z.namelist():
                if (
                    k.startswith(("dependencies/", "paper-tools/", "model/"))
                    or k == "sources/loopqwen-research.bundle"
                ):
                    z.extract(k, work)
        git = tmp / "restored.git"
        command(
            "git",
            "clone",
            "--quiet",
            "--bare",
            work / "sources/loopqwen-research.bundle",
            git,
        )
        command(
            "git",
            "-C",
            git,
            "fetch",
            "--quiet",
            work / "sources/update.bundle",
            "refs/heads/main:refs/heads/main",
        )
        assert (
            subprocess.check_output(
                ["git", "-C", git, "rev-parse", "main"], text=True
            ).strip()
            == revision
        )
        for p, k in (
            ("paper/loopqwen-study.pdf", "paper/loopqwen-study.pdf"),
            (
                "research/results/" + NAME + "/cross-view.safetensors",
                "policy/cross-view.safetensors",
            ),
        ):
            assert (
                subprocess.check_output(["git", "-C", git, "show", revision + ":" + p])
                == (work / k).read_bytes()
            )
        venv = tmp / "env"
        python = venv / "bin/python"
        command(
            "uv", "venv", "--no-project", "--offline", "--python", sys.executable, venv
        )
        command(
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
            work / "runtime" / wheel,
            *sorted((work / "paper-tools").glob("*.whl")),
            *sorted((work / "figure-tools").glob("*.whl")),
        )
        command("uv", "pip", "check", "--python", python)
        cache = tmp / "empty"
        cache.mkdir()
        env = dict(
            os.environ,
            HF_HOME=str(cache),
            HF_HUB_CACHE=str(cache / "hub"),
            HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            MPLCONFIGDIR=str(tmp / "matplotlib"),
        )
        site = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
        command(
            python,
            "-I",
            "-c",
            NETWORK_GUARD
            + "import runpy;runpy.run_module('loopqwen.weight_transport',run_name='__main__')",
            "unpack",
            "model/tool-language-packed-05.lqzip",
            "release-05",
            cwd=work,
            env=env,
            stdout=subprocess.DEVNULL,
        )
        paper_code = "from pathlib import Path;import runpy; e=runpy.run_path('scripts/assemble_evidence_paper.py');e['edition'](Path('paper'),'cross-view',verify=True)"
        command(python, "-I", "-c", NETWORK_GUARD + paper_code, cwd=work, env=env)
        analysis = tmp / "analysis"
        command("git", "clone", "--quiet", "--local", "--branch", "main", git, analysis)
        (analysis / "exports").mkdir(exist_ok=True)
        (analysis / "exports/tool-language-packed-05").symlink_to(work / "release-05")
        external = analysis / "data/research_external"
        external.mkdir(parents=True, exist_ok=True)
        for rel in ("squad-train-v1.1.json", "quoref/train-dev.zip"):
            (external / rel).parent.mkdir(parents=True, exist_ok=True)
            (external / rel).symlink_to(work / "sources" / rel)
        generated = [
            name
            for name in hashes
            if name.startswith("paper/cross-view")
            and (
                name.endswith(
                    (".csv", ".svg", ".json", ".pdf", "-metrics.typ", "-intervals.typ")
                )
            )
        ] + ["paper/loopqwen-study.pdf"]
        for name in generated:
            (analysis / name).unlink()
        modules = (
            "cross_view_material",
            "cross_view_source_check",
            "cross_view_exposure",
            "cross_view_reading",
            "paper_cross_view",
        )
        for module in modules:
            code = (
                "import runpy,sys;from pathlib import Path;sys.path.append(str(Path.cwd()));"
                "import loopqwen;assert Path(loopqwen.__file__).is_relative_to(sys.prefix);"
                f"runpy.run_module('scripts.{module}',run_name='__main__')"
            )
            command(python, "-I", "-c", NETWORK_GUARD + code, cwd=analysis, env=env)
        for name in generated:
            assert digest(analysis / name) == hashes[name], name
        results = []
        for key, i, ident in cases:
            args = read(work / "examples" / (key + ".json"))
            got = json.loads(
                subprocess.check_output(
                    [python, "-I", "-c", NETWORK_GUARD + CODE, site, *args],
                    env=env,
                    cwd=work,
                )
            )
            assert (
                got["initial_prompt_identical"]
                and got["base_source"] == "certified_release_text"
            )
            tr = {}
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                r = unpack(work / "policy" / (arm + ".json.gz"))[i]
                orig = [trace(q["trajectory"]) for q in r["requests"]]
                exact = [trace(q) for q in got["runs"][side]]
                assert r["id"] == ident and orig == exact
                tr[side] = fingerprint(exact)
            results.append({"case": key, "id": ident, "traces": tr})
        assert not any(f.is_file() for f in cache.rglob("*"))
        receipt = {
            "name": ZIP.name,
            "sha256": digest(archive),
            "bytes": archive.stat().st_size,
            "source_revision": revision,
            "wheel_source": subprocess.check_output(
                ["git", "rev-parse", WHEEL], text=True, cwd=ROOT
            ).strip(),
            "payloads": len(hashes),
            "required_kit": kit["sha256"],
            "outside_checkout": True,
            "network_blocked": True,
            "empty_model_cache": True,
            "history_restored": True,
            "teacher_checkpoint_present": False,
            "teacher_targets": digest(work / "policy/teacher.json.gz"),
            "compiled_layer": digest(work / "policy/cross-view.safetensors"),
            "verified_cases": results,
            "paper_sha256": digest(work / "paper/loopqwen-study.pdf"),
            "regenerated_from_original_traces": sorted(generated),
        }
        if ZIP.exists():
            assert digest(ZIP) == receipt["sha256"]
        else:
            partial = ZIP.with_suffix(".tmp")
            shutil.copyfile(archive, partial)
            partial.replace(ZIP)
        if RECEIPT.exists():
            assert read(RECEIPT) == receipt
        else:
            save(RECEIPT, receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
