"""Assemble an offline native four-layer checkpoint and an independently optional certified Base."""

import argparse
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

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.ordinary_model import CERTIFIED_INVENTORY
from loopqwen.reentry import PACKED, packed_path

CODE_REV = "55b2fab9be1c65a34771ed7b5d347f7262ff8aa5"
RESULT = ROOT / "research/results/table-reentry-fit-01"
EXPORT = ROOT / "exports"
MODEL = EXPORT / "loopqwen-reentry-model-01.zip"
REFERENCE = EXPORT / "loopqwen-reentry-reference-01.zip"
RECEIPT = ROOT / "docs/measurements/reentry-distribution.json"


def verify_members(archive, hashes, checksum_name):
    import hashlib

    with ZipFile(archive) as z:
        assert set(z.namelist()) == {checksum_name, *hashes}
        for name, sha in hashes.items():
            with z.open(name) as member:
                assert hashlib.file_digest(member, "sha256").hexdigest() == sha
        lines = z.read(checksum_name).decode().splitlines()
        assert dict(line.split("  ", 1)[::-1] for line in lines) == hashes


def save_archive(path, files, *, checksum_name="SHA256SUMS"):
    with TemporaryDirectory(dir=path.parent) as tmp:
        output = Path(tmp) / "kit.zip"
        hashes = write_archive(output, files, checksum_name=checksum_name)
        verify_members(output, hashes, checksum_name)
        sha = digest(output)
        if path.is_file():
            assert digest(path) == sha
        else:
            output.replace(path)
    return {
        "archive": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha,
        "members": hashes,
    }


def assemble():
    archive = read(RESULT / "checksums.json")
    spec = read(RESULT / "registration.json")
    fit = read(RESULT / "fit.json")
    root = ROOT / spec["checkpoint"]
    inventory = root / "tool-export-files.json"
    assert digest(inventory) == CERTIFIED_INVENTORY
    source = read(inventory)
    assert fit["backbone"]["sha256"] == digest(packed_path(root))
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        wheel = build(work / "one", CODE_REV)
        assert digest(wheel) == digest(build(work / "two", CODE_REV))
        files = {
            "README.md": ROOT / "docs/REENTRY_MODEL.md",
            "runtime/" + wheel.name: wheel,
            "runtime/uv.lock": ROOT / "uv.lock",
            "runtime/pyproject.toml": ROOT / "pyproject.toml",
            "runtime/LICENSE": ROOT / "LICENSE",
            "paper/reentry-fit.pdf": ROOT / "paper/reentry-fit.pdf",
            "reentry-backbone-01/" + PACKED: packed_path(root),
        }
        for name in (
            "config.json",
            "provenance.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "tool/tool-model.json",
            "LICENSE",
            *(k for k in source if k.startswith("chat_templates/")),
        ):
            if name in source:
                assert source[name] == digest(root / name)
                files["checkpoint/" + name] = root / name
        for name in (
            "registration.json",
            "fit.json",
            "reentry.safetensors",
            "decision.json",
            "checksums.json",
        ):
            if name != "checksums.json":
                assert digest(RESULT / name) == archive[name]
            files["policy/" + name] = RESULT / name
        training = read(ROOT / spec["training"])["examples"]
        question = next(
            r
            for r in training
            if r["id"] == "database/cosql-b34990e8aea3fd41ca46-database_schema"
        )
        files["examples/database-question.txt"] = question["messages"][0][
            "content"
        ].encode()
        files["examples/source.md"] = (
            ROOT / "research/sources/database-dialogues/README.md"
        )
        db = next(
            r
            for r in read(ROOT / "research/sources/database-dialogues/originals.json")[
                "cohorts"
            ]["train"]
            if r["database"] == "customers_card_transactions"
        )
        with ZipFile(ROOT / "data/research_external/cosql/cosql.zip") as upstream:
            example = work / (db["database"] + ".sqlite")
            example.write_bytes(upstream.read(db["member"]))
        assert digest(example) == db["sqlite_sha256"]
        files["examples/" + example.name] = example
        a = save_archive(MODEL, files)
        reference = root / "model.safetensors"
        assert source[reference.name] == fit["backbone"]["base"] == digest(reference)
        ref = {
            "checkpoint/model.safetensors": reference,
            "checkpoint/tool-export-files.json": inventory,
        }
        b = save_archive(REFERENCE, ref, checksum_name="reference/SHA256SUMS")
        record = {
            "runtime_revision": CODE_REV,
            "wheel_sha256": digest(wheel),
            "weights": fit["weights"],
            "backbone": fit["backbone"]["sha256"],
            "source_archive": digest(RESULT / "checksums.json"),
            "recurrent": a,
            "reference": b,
        }
        if RECEIPT.is_file():
            assert read(RECEIPT) == record
        else:
            save(RECEIPT, record)
        print(
            a["bytes"],
            b["bytes"],
            "bytes in verified independent model and reference archives",
            flush=True,
        )


def installed():
    report = read(RECEIPT)
    assert (
        digest(MODEL) == report["recurrent"]["sha256"]
        and digest(REFERENCE) == report["reference"]["sha256"]
    )
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        with ZipFile(MODEL) as z:
            z.extractall(work)
        assert not (work / "checkpoint/model.safetensors").exists()
        (wheel,) = (work / "runtime").glob("*.whl")
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
                str(wheel),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        env = {
            **os.environ,
            "HF_HOME": str(work / "empty-cache"),
            "HF_HUB_CACHE": str(work / "empty-cache/hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        cmd = [
            sys.executable,
            "-I",
            "-c",
            NETWORK_GUARD + CODE,
            str(site),
            "ask",
            "--checkpoint",
            "checkpoint",
            "--session-policy",
            "policy",
            "--tool-format",
            "qwen",
            "--trace",
        ]
        args = [
            "--relational-db",
            "examples/customers_card_transactions.sqlite",
            "--prompt-file",
            "examples/database-question.txt",
            "--max-turns",
            "3",
            "--max-tokens",
            "128",
        ]

        def execute(extra):
            return json.loads(
                subprocess.check_output([*cmd, *extra], cwd=work, env=env)
            )

        standalone = execute(args)
        assert (
            standalone["turns"][0]["prompt_ids"]
            == read(RESULT / "trained-parity.json")["ids"]
        )
        with ZipFile(REFERENCE) as z:
            z.extractall(work)
        full = execute([*args, "--compare-base"])
        (candidate,) = full["runs"]["recurrent"]
        (base,) = full["runs"]["base"]
        assert (
            full["initial_prompt_identical"]
            and full["base_source"] == "certified_release_text"
        )
        assert (
            trace(candidate) == trace(standalone)
            and base["turns"][0]["prompt_ids"] == candidate["turns"][0]["prompt_ids"]
        )
        raw_question = (
            "The library stores one atlas on each shelf.\n" * 25
        ) + "How many atlases are on two occupied shelves?"
        check = execute(
            [
                "--prompt",
                raw_question,
                "--compare-base",
                "--cache-state-prefill",
                "--max-turns",
                "1",
                "--max-tokens",
                "8",
            ]
        )
        for name, rows in check["runs"].items():
            (r,) = rows
            (turn,) = r["turns"]
            assert (
                r["initial_tokens"] > 128
                and turn["cache_state_blocks"] == (r["initial_tokens"] - 1) // 128
            )
            expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
            if name == "recurrent":
                expected.add((23, 1))
            assert {(h["layer"], h["turn"]) for h in turn["kv_history"]} == expected
            assert all(
                h["offset"]
                == h["retained"]
                == r["initial_tokens"] + len(turn["token_ids"]) - 1
                for h in turn["kv_history"]
            )
        assert not any(p.is_file() for p in (work / "empty-cache").rglob("*"))
        # Timing is a property of this installation verification and is kept in its individual traces.
        examples = {
            "standalone_database": standalone,
            "database_pair": full,
            "plain_pair": check,
        }
        from loopqwen.evaluation.mechanism import packed

        output = ROOT / "docs/measurements/reentry-installed-traces.json.gz"
        result = {
            "distribution": digest(RECEIPT),
            "initial_database_tokens": len(standalone["turns"][0]["prompt_ids"]),
            "database_trajectories": {
                n: fingerprint(trace(v[0])) for n, v in full["runs"].items()
            },
            "plain_trajectories": {
                n: fingerprint(trace(v[0])) for n, v in check["runs"].items()
            },
            "standalone_matches_paired": True,
            "base_loaded_offline": True,
            "empty_model_cache": True,
        }
        receipt = ROOT / "docs/measurements/reentry-installed.json"
        if receipt.is_file():
            assert {k: v for k, v in read(receipt).items() if k != "traces"} == result
            assert digest(output) == read(receipt)["traces"]
        else:
            packed(output, examples)
            save(receipt, {**result, "traces": digest(output)})
        print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["assemble", "installed"])
    if p.parse_args().action == "assemble":
        assemble()
    else:
        installed()
