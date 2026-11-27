"""Build a research-policy add-on and independently exercise the installed model and paper."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Edition:
    study: str
    archive: str
    manual: str
    paper: str
    verification: str
    bundle: str
    modules: tuple
    cohort: str = "01"

    @property
    def study_id(self):
        return self.study + "-" + self.cohort


PROOF = Edition(
    "proof-curriculum",
    "loopqwen-proof-update-05.zip",
    "PROOF_UPDATE.md",
    "loopqwen-proof-study.pdf",
    "proof-curriculum-field-update-05.json",
    "proof-study.bundle",
    (
        ("proof_curriculum_exposure", ()),
        ("proof_curriculum_reading", ("report",)),
        ("paper_proof_curriculum", ()),
    ),
)
FIELD = Edition(
    "field-selection",
    "loopqwen-field-selection-06.zip",
    "FIELD_SELECTION.md",
    "loopqwen-field-study.pdf",
    "field-selection-update-06.json",
    "field-selection.bundle",
    (
        ("verify_field_selection_frozen", ()),
        ("field_selection_provenance", ("--compact",)),
        ("field_selection_oracle", ()),
        ("field_selection_exposure", ()),
        ("field_selection_analysis", ("report",)),
        ("field_selection_scoring_audit", ()),
        ("paper_field_selection", ()),
    ),
)


VALID = Edition(
    "valid-return",
    "loopqwen-valid-return-07.zip",
    "VALID_RETURN.md",
    "loopqwen-preference-study.pdf",
    "valid-return-update-07.json",
    "valid-return.bundle",
    (
        (
            "verify_field_selection_frozen",
            ("--root", "research/results/valid-return-01"),
        ),
        ("field_selection_provenance", ("--compact", "--valid-return")),
        ("agenda_sources", ()),
        ("valid_return_oracle", ()),
        ("valid_return_exposure", ()),
        ("valid_return_analysis", ("report",)),
        ("valid_return_ballots", ()),
        ("valid_return_batches", ()),
        ("paper_valid_return", ()),
    ),
)


LINKED = Edition(
    "linked-rollin",
    "loopqwen-linked-rollin-08.zip",
    "LINKED_ROLLIN.md",
    "loopqwen-grounding-study.pdf",
    "linked-rollin-update-08.json",
    "linked-rollin.bundle",
    (
        ("verify_linked_sources", ()),
        ("linked_rollin_exposure", ()),
        ("linked_rollin_analysis", ("report",)),
        ("paper_cache_state", ()),
        ("paper_linked_rollin", ()),
    ),
    cohort="02",
)

FOLLOWUP = Edition(
    "followup-protected",
    "loopqwen-followup-09.zip",
    "FOLLOWUP_STUDY.md",
    "loopqwen-followup-study.pdf",
    "followup-update-09.json",
    "followup.bundle",
    (
        ("followup_oracle", ()),
        ("research_archive", ("followup-protected-01", "--verify")),
        ("paper_followup", ()),
    ),
)


def tracked(revision, path):
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", revision, "--", path],
        cwd=ROOT,
    )
    return [ROOT / n.decode() for n in names.split(b"\0") if n]


def make(tmp, revision, update, edition):
    policy = ROOT / "research/results" / edition.study_id
    spec = read(policy / "registration.json")
    panel = read(ROOT / spec["material"])
    docs = ROOT / "paper"
    prefix = "followup" if edition == FOLLOWUP else edition.study
    files = {
        "README.md": ROOT / "docs" / edition.manual,
        "sources/" + edition.bundle: history_update(
            tmp, revision, update["source_revision"]
        ),
    }
    files.update(
        {
            "policy/" + p.name: p
            for p in tracked(revision, str(policy.relative_to(ROOT)))
        }
    )
    files.update(
        {
            "paper/" + p.name: p
            for p in tracked(revision, "paper")
            if p.name.startswith(prefix + "-") or p.name.startswith(prefix + ".")
        }
    )
    files["paper/" + edition.paper] = docs / edition.paper
    for p in tracked(revision, "docs/measurements"):
        if p.name.startswith(prefix + "-") and p.name != edition.verification:
            files["verification/" + p.name] = p
    files["sources/REASONED-EVIDENCE-SOURCES.md"] = (
        ROOT / "research/licenses/REASONED-EVIDENCE-SOURCES.md"
    )
    if edition in (FIELD, VALID):
        for wheel in panel["wheels"]:
            source = (
                ROOT / "data/research_external/field_selection/wheels" / wheel["wheel"]
            )
            assert digest(source) == wheel["wheel_sha256"]
            files["source-wheels/" + wheel["wheel"]] = source.read_bytes()
    if edition == VALID:
        from agenda_sources import FILES, source

        for year, formats in FILES.items():
            for fmt in formats:
                files[f"schedule-sources/{year}.{fmt}"] = source(
                    year, fmt, False
                ).read_bytes()
    if edition == LINKED:
        files["paper/cache-state-conversations.svg"] = (
            docs / "cache-state-conversations.svg"
        )
        for source in tracked(revision, "research/sources/linked-tables"):
            files["linked-sources/" + source.name] = source
    if edition == FOLLOWUP:
        from scripts.followup_oracle import SHA, UPSTREAM
        from scripts.wheel_archive_sources import CACHE

        runtime = read(ROOT / "docs/measurements/followup-runtime.json")
        installed_wheel = ROOT / runtime["archive"]
        assert digest(installed_wheel) == runtime["sha256"] and digest(UPSTREAM) == SHA
        files["updated-runtime/" + installed_wheel.name] = installed_wheel.read_bytes()
        files["source-scorer/" + UPSTREAM.name] = UPSTREAM.read_bytes()
        cohorts = read(ROOT / "research/sources/wheel-archives/enrollment.json")[
            "cohorts"
        ]
        for cohort in cohorts.values():
            for item in cohort:
                archive = CACHE / "wheels" / item["file"]
                assert digest(archive) == item["sha256"]
                files["source-wheels/" + archive.name] = archive.read_bytes()
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
    if edition == VALID:
        event = next(w["cluster"] for w in rows if w["family"] == "agenda")
        selected.extend(w for w in rows if w["cluster"] == event)
    if edition == LINKED:
        first = next(w["cluster"] for w in rows if w["family"] == "linked_inline")
        event = next(w["cluster"] for w in rows if w["family"] == "agenda")
        selected = [w for w in rows if w["cluster"] in (first, event, pair)]
        selected.append(next(w for w in rows if w["family"] == "retention"))
    if edition == FOLLOWUP:
        clusters = {
            next(w["cluster"] for w in rows if w["family"] == family)
            for family in ("human_first", "archive", "agenda")
        }
        selected = [w for w in rows if w["cluster"] in clusters]
        selected.append(next(w for w in rows if w["family"] == "retention"))
    assert len(selected) == (7 if edition in (VALID, LINKED, FOLLOWUP) else 5)
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
        paths = (
            w["files"]
            if "files" in w
            else ([w["document"]] if w.get("document") else [])
        )
        for path in paths:
            name = "examples/" + path
            files[name] = panel["documents"][path].encode()
            args.extend(["--keyed-csv", name])
        if w["family"] == "agenda":
            args.extend(["--resource", "calendar"])
        for wheel in w.get("wheels", ()):
            args.extend(["--wheel", "source-wheels/" + wheel])
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


def main(edition=PROOF):
    dest = ROOT / "exports" / edition.archive
    receipt_path = ROOT / "docs/measurements" / edition.verification
    base, update = (
        read(ROOT / "docs/measurements" / n)
        for n in ("field-kit-01.json", "cross-view-field-update-04.json")
    )
    revision = (
        read(receipt_path)["source_revision"]
        if receipt_path.exists()
        else subprocess.check_output(
            (
                ["git", "log", "-1", "--format=%H", "--", "paper/" + edition.paper]
                if edition in (FIELD, VALID, LINKED, FOLLOWUP)
                else ["git", "rev-parse", "HEAD"]
            ),
            cwd=ROOT,
            text=True,
        ).strip()
    )
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        work = tmp / "installed"
        work.mkdir()
        files, cases = make(tmp, revision, update, edition)
        archive = tmp / dest.name
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
        for name in ("update.bundle", edition.bundle):
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
        command(
            "git", "clone", "--quiet", "--local", "--branch", "main", repo, checkout
        )
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
            *sorted(
                (work / ("updated-runtime" if edition == FOLLOWUP else "runtime")).glob(
                    "*.whl"
                )
            ),
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
                == (
                    checkout / "research/results" / edition.study_id / p.name
                ).read_bytes()
            )
        generated = [
            k
            for k in hashes
            if k.startswith("paper/") and k.endswith((".csv", ".svg", ".pdf", ".json"))
        ]
        for n in generated:
            (checkout / n).unlink()
        if edition in (FIELD, VALID):
            wheel_cache = checkout / "data/research_external/field_selection/wheels"
            wheel_cache.parent.mkdir(parents=True, exist_ok=True)
            wheel_cache.symlink_to(work / "source-wheels")
        if edition == VALID:
            (checkout / "data/research_external/fosdem").symlink_to(
                work / "schedule-sources"
            )
        if edition == FOLLOWUP:
            external = checkout / "data/research_external"
            cache_path = external / "wheel-archives"
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "wheels").symlink_to(work / "source-wheels")
            (external / "quac-scorer.py").symlink_to(
                work / "source-scorer/quac-scorer.py"
            )
        for module, args in edition.modules:
            research_path = (
                "sys.path.insert(0,str(Path.cwd()));"
                if edition in (VALID, LINKED, FOLLOWUP)
                or module == "verify_field_selection_frozen"
                else "sys.path.append(str(Path.cwd()));"
            )
            code = (
                NETWORK_GUARD
                + "import sys,runpy;from pathlib import Path;"
                + research_path
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
            "name": dest.name,
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
        if dest.exists():
            assert digest(dest) == receipt["sha256"]
        else:
            partial = dest.with_suffix(".tmp")
            shutil.copyfile(archive, partial)
            partial.replace(dest)
        if receipt_path.exists():
            assert receipt == read(receipt_path)
        else:
            save(receipt_path, receipt)
        print(
            "Verified",
            len(results),
            "installed direct pairs and",
            len(generated),
            "rebuilt paper artifacts",
            receipt["sha256"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    editions = parser.add_mutually_exclusive_group()
    editions.add_argument("--field-selection", action="store_true")
    editions.add_argument("--valid-return", action="store_true")
    editions.add_argument("--linked-rollin", action="store_true")
    editions.add_argument("--followup", action="store_true")
    options = parser.parse_args()
    main(
        FOLLOWUP
        if options.followup
        else LINKED
        if options.linked_rollin
        else VALID
        if options.valid_return
        else FIELD
        if options.field_selection
        else PROOF
    )
