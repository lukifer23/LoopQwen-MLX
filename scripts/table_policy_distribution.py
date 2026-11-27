"""Ship the actual measured policy and replay complete original/altered and plain requests offline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portable_base_wheel import NETWORK_GUARD
from reentry_distribution import (
    MODEL,
    REFERENCE,
    save_archive,
    verify_members,
)
from reentry_distribution import (
    RECEIPT as PARENT,
)
from verify_runtime_wheel import CODE, ROOT, trace

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE, mount
from scripts.table_policy_reading import RATINGS, committed

PRIMARY = ROOT / "research/results/table-policy-gradient-01"
FIT = ROOT / "research/results/table-execution-learning-01"
DEST = ROOT / "exports/loopqwen-table-policy-01.zip"
DOC = ROOT / "docs/measurements"
RECEIPT = DOC / "table-policy-distribution.json"


def cases():
    rows = read(ROOT / "research/data/table-policy-panel.json")["rows"]
    worlds = read(SOURCE)["cohorts"]["dev"]
    w = min(worlds, key=lambda r: fingerprint([8190, "installed-table", r["id"]]))
    rr = sorted(
        (r for r in rows if r["family"] == "changed_table" and r["row_id"] == w["id"]),
        key=lambda r: r["world"] != 0,
    )
    assert (
        len(rr) == 2
        and rr[0]["initial_ids"] == rr[1]["initial_ids"]
        and rr[0]["gold"] != rr[1]["gold"]
    )
    plain = min(
        (r for r in rows if r["family"] == "open"),
        key=lambda r: fingerprint([8190, "installed-text", r["id"]]),
    )
    return w, [*rr, plain]


def assemble():
    committed(RATINGS)
    evidence = read(PRIMARY / "training-parent.json")
    assert evidence["weights"] == digest(FIT / "reentry.safetensors") and evidence[
        "archive"
    ] == digest(FIT / "checksums.json")
    w, rr = cases()
    files = {
        "table-policy/README.md": ROOT / "docs/TABLE_POLICY.md",
        "table-policy/paper/table-policy.pdf": ROOT / "paper/table-policy.pdf",
        "table-policy/paper/table-policy-learning.pdf": ROOT
        / "paper/table-policy-learning.pdf",
        "table-policy/paper/reentry-fit.pdf": ROOT / "paper/reentry-fit.pdf",
        "table-policy/examples/question.txt": rr[0]["questions"][0].encode(),
        "table-policy/examples/open.txt": rr[-1]["questions"][0].encode(),
        "table-policy/examples/source.json": (
            json.dumps(w, ensure_ascii=False, indent=2) + "\n"
        ).encode(),
        "table-policy/examples/source.md": ROOT
        / "research/sources/changed-tables/README.md",
    }
    for name in (
        "registration.json",
        "fit.json",
        "checksums.json",
        "decision.json",
        "reentry.safetensors",
    ):
        files["table-policy/policy/" + name] = FIT / name
    for name in (
        "decision.json",
        "checksums.json",
        "registration.json",
        "training-parent.json",
    ):
        files["table-policy/comparison/" + name] = PRIMARY / name
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for r, label in zip(rr[:2], ("original", "altered"), strict=True):
            v = next(v for v in w["versions"] if v["world"] == r["world"])
            files["table-policy/examples/" + label + "/table.sqlite"] = mount(
                folder / label, v["rows"]
            )
        rec = save_archive(DEST, files, checksum_name="table-policy/SHA256SUMS")
    record = {
        "archive": rec,
        "policy": evidence["weights"],
        "backbone": read(FIT / "fit.json")["backbone"]["sha256"],
        "required_model": read(PARENT)["recurrent"]["sha256"],
        "optional_base": read(PARENT)["reference"]["sha256"],
        "requests": [r["id"] for r in rr],
        "primary_archive": digest(PRIMARY / "checksums.json"),
    }
    if RECEIPT.is_file():
        assert read(RECEIPT) == record
    else:
        save(RECEIPT, record)
    print(rec["bytes"], rec["sha256"])


def installed():
    rec = read(RECEIPT)
    for archive, sha in (
        (MODEL, rec["required_model"]),
        (REFERENCE, rec["optional_base"]),
        (DEST, rec["archive"]["sha256"]),
    ):
        assert digest(archive) == sha
    verify_members(DEST, rec["archive"]["members"], "table-policy/SHA256SUMS")
    _, rr = cases()
    expected = {
        out: {
            r["id"]: r["requests"][0]["trajectory"]
            for r in unpack(PRIMARY / (side + ".json.gz"))
        }
        for out, side in (("recurrent", "candidate"), ("base", "control"))
    }
    reports = {}
    fingerprints = {}
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        for path in (MODEL, DEST):
            with ZipFile(path) as z:
                z.extractall(work)
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
            "table-policy/policy",
            "--tool-format",
            "qwen",
            "--trace",
            "--max-turns",
            "3",
            "--max-tokens",
            "384",
        ]

        def run(extra):
            return json.loads(
                subprocess.check_output([*cmd, *extra], cwd=work, env=env)
            )

        original = [
            "--relational-db",
            "table-policy/examples/original/table.sqlite",
            "--prompt-file",
            "table-policy/examples/question.txt",
        ]
        assert not (work / "checkpoint/model.safetensors").exists()
        alone = run(original)
        assert trace(alone) == trace(expected["recurrent"][rr[0]["id"]])
        with ZipFile(REFERENCE) as z:
            z.extractall(work)
        reports["standalone"] = alone
        for r, options in zip(
            rr,
            (
                original,
                [v.replace("examples/original", "examples/altered") for v in original],
                ["--prompt-file", "table-policy/examples/open.txt"],
            ),
            strict=True,
        ):
            pair = run([*options, "--compare-base"])
            fingerprints[r["id"]] = {}
            for arm, records in pair["runs"].items():
                (actual,) = records
                assert actual["turns"][0]["prompt_ids"] == r["initial_ids"]
                assert trace(actual) == trace(expected[arm][r["id"]]), (r["id"], arm)
                fingerprints[r["id"]][arm] = fingerprint(trace(actual))
            reports[r["id"]] = pair
        assert not any(p.is_file() for p in (work / "empty-cache").rglob("*"))
    path = DOC / "table-policy-installed.json"
    raw = DOC / "table-policy-installed-traces.json.gz"
    receipt = {
        "package": rec["archive"]["sha256"],
        "policy": rec["policy"],
        "offline": True,
        "no_cached_models": True,
        "standalone_verified": True,
        "primary_trajectories": fingerprints,
    }
    if path.exists():
        assert {
            k: v for k, v in read(path).items() if k != "traces"
        } == receipt and digest(raw) == read(path)["traces"]
    else:
        packed(raw, reports)
        save(path, {**receipt, "traces": digest(raw)})
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["assemble", "installed"])
    args = parser.parse_args()
    if args.mode == "assemble":
        assemble()
    else:
        installed()
