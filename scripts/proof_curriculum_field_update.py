"""Build a research-policy add-on and independently exercise the installed model and paper."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from cross_view_field_update import command
from field_binding_update import history_update
from field_kit import write_archive
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT, trace

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

NAME = "proof-curriculum-01"
DEST = ROOT / "exports/loopqwen-proof-update-05.zip"
RECEIPT = ROOT / "docs/measurements/proof-curriculum-field-update-05.json"


def make(tmp, revision, update):
    policy = ROOT / "research/results" / NAME
    spec = read(policy / "registration.json")
    panel = read(ROOT / spec["material"])
    docs = ROOT / "paper"
    files = {
        "README.md": ROOT / "docs/PROOF_UPDATE.md",
        "sources/proof-study.bundle": history_update(
            tmp, revision, update["source_revision"]
        ),
    }
    files.update({"policy/" + p.name: p for p in policy.iterdir()})
    files.update({"paper/" + p.name: p for p in docs.glob("proof-curriculum*")})
    files["paper/loopqwen-proof-study.pdf"] = docs / "loopqwen-proof-study.pdf"
    for p in (ROOT / "docs/measurements").glob("proof-curriculum-*"):
        if p != RECEIPT:
            files["verification/" + p.name] = p
    files["sources/REASONED-EVIDENCE-SOURCES.md"] = (
        ROOT / "research/licenses/REASONED-EVIDENCE-SOURCES.md"
    )
    rows = panel["rows"]
    # Select by enrollment order alone: the first story through both channels, a whole
    # practical contrast, and first retained table. No success or latency is consulted.
    original = next(w for w in rows if w["family"] == "inline")
    pair = next(w["cluster"] for w in rows if w["family"] == "wheel")
    selected = [
        w
        for w in rows
        if w["id"] in (original["id"], original["id"].removesuffix("-inline") + "-file")
        or w["cluster"] == pair
    ]
    selected.append(next(w for w in rows if w["family"] == "retention"))
    assert len(selected) == 5
    cases = []
    for j, w in enumerate(selected):
        key = f"{j + 1}-{w['family']}"
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
            name = "examples/" + w["document"]
            files[name] = panel["documents"][w["document"]].encode()
            args.extend(["--keyed-csv", name])
        files["examples/" + key + ".json"] = (
            json.dumps(args, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        cases.append((key, rows.index(w)))
    for name, p in list(files.items()):
        if isinstance(p, Path) and p.is_relative_to(ROOT):
            files[name] = subprocess.check_output(
                ["git", "show", revision + ":" + p.relative_to(ROOT).as_posix()],
                cwd=ROOT,
            )
    return files, cases


def main():
    base, update = (
        read(ROOT / "docs/measurements" / n)
        for n in ("field-kit-01.json", "cross-view-field-update-04.json")
    )
    revision = (
        read(RECEIPT)["source_revision"]
        if RECEIPT.exists()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        work = tmp / "installed"
        work.mkdir()
        files, cases = make(tmp, revision, update)
        archive = tmp / DEST.name
        hashes = write_archive(archive, files)
        with ZipFile(archive) as z:
            z.extractall(work)
        assert all(digest(work / k) == v for k, v in hashes.items())
        for parent, prefixes, names in (
            (
                base,
                ("dependencies/", "model/", "paper-tools/"),
                {"sources/loopqwen-research.bundle"},
            ),
            (update, ("runtime/", "figure-tools/"), {"sources/update.bundle"}),
        ):
            p = ROOT / "exports" / parent["name"]
            assert digest(p) == parent["sha256"]
            with ZipFile(p) as z:
                for n in z.namelist():
                    if n.startswith(prefixes) or n in names:
                        z.extract(n, work)
        repo = tmp / "restored.git"
        command(
            "git",
            "clone",
            "--quiet",
            "--bare",
            work / "sources/loopqwen-research.bundle",
            repo,
        )
        for name in ("update.bundle", "proof-study.bundle"):
            command(
                "git",
                "-C",
                repo,
                "fetch",
                "--quiet",
                work / "sources" / name,
                "refs/heads/main:refs/heads/main",
            )
        assert (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "main"], text=True
            ).strip()
            == revision
        )
        checkout = tmp / "research"
        command("git", "clone", "--quiet", "--local", repo, checkout)
        python = tmp / "venv/bin/python"
        command(
            "uv",
            "venv",
            "--offline",
            "--no-project",
            "--python",
            sys.executable,
            python.parent.parent,
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
            *sorted((work / "runtime").glob("*.whl")),
            *sorted((work / "paper-tools").glob("*.whl")),
            *sorted((work / "figure-tools").glob("*.whl")),
        )
        command("uv", "pip", "check", "--python", python)
        cache = tmp / "empty-model-cache"
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
        (checkout / "exports").mkdir(exist_ok=True)
        (checkout / "exports/tool-language-packed-05").symlink_to(work / "release-05")
        for p in (work / "policy").iterdir():
            assert (
                p.read_bytes()
                == (checkout / "research/results" / NAME / p.name).read_bytes()
            )
        generated = [
            k
            for k in hashes
            if k.startswith("paper/") and k.endswith((".csv", ".svg", ".pdf", ".json"))
        ]
        for n in generated:
            (checkout / n).unlink()
        for module, args in (
            ("proof_curriculum_exposure", []),
            ("proof_curriculum_reading", ["report"]),
            ("paper_proof_curriculum", []),
        ):
            code = (
                NETWORK_GUARD
                + "import sys,runpy;from pathlib import Path;sys.path.append(str(Path.cwd()));"
                + f"runpy.run_module('scripts.{module}',run_name='__main__')"
            )
            command(python, "-I", "-c", code, *args, cwd=checkout, env=env)
        assert all(digest(checkout / n) == hashes[n] for n in generated)
        site = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
        results = []
        for key, index in cases:
            args = read(work / "examples" / (key + ".json"))
            got = json.loads(
                subprocess.check_output(
                    [python, "-I", "-c", NETWORK_GUARD + CODE, site, *args],
                    cwd=work,
                    env=env,
                )
            )
            assert (
                got["initial_prompt_identical"]
                and got["base_source"] == "certified_release_text"
            )
            streams = {}
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                r = unpack(work / "policy" / (arm + ".json.gz"))[index]
                orig = [trace(q["trajectory"]) for q in r["requests"]]
                actual = [trace(q) for q in got["runs"][side]]
                assert actual == orig
                streams[side] = fingerprint(actual)
            results.append({"example": key, "id": r["id"], "traces": streams})
        assert not any(p.is_file() for p in cache.rglob("*"))
        receipt = {
            "name": DEST.name,
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
            "payloads": len(hashes),
            "requires": {v["name"]: v["sha256"] for v in (base, update)},
            "source_revision": revision,
            "history_restored": True,
            "network_blocked": True,
            "empty_model_cache": True,
            "isolated_installed_cli": True,
            "policy_weights": hashes["policy/cross-view.safetensors"],
            "actual_comparisons": results,
            "regenerated": {n: hashes[n] for n in generated},
        }
        if DEST.exists():
            assert digest(DEST) == receipt["sha256"]
        else:
            partial = DEST.with_suffix(".tmp")
            shutil.copyfile(archive, partial)
            partial.replace(DEST)
        if RECEIPT.exists():
            assert receipt == read(RECEIPT)
        else:
            save(RECEIPT, receipt)
        print(
            "Verified",
            len(results),
            "installed direct pairs and",
            len(generated),
            "rebuilt paper artifacts",
            receipt["sha256"],
        )


if __name__ == "__main__":
    main()
