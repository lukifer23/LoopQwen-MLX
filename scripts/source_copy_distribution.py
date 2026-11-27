"""Package the actual learned model and verify bit-identical traces under an offline installed wheel."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reentry_distribution import save_archive, verify_members
from verify_runtime_wheel import build, trace, CODE
from portable_base_wheel import NETWORK_GUARD
from loopqwen.conditional_reentry import PACKED
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.ordinary_model import CERTIFIED_INVENTORY
from scripts.conversational_reasoning_sources import CACHE, OUT as READING
from scripts.register_source_copy import RESULT, FIT
from scripts.table_policy_material import persist

REV = "7cfd019"
GUIDE_REV = "334553039bee273e8637a8272ecf1866b6cec6cb"
MODEL = ROOT / "exports/loopqwen-source-copy-01.zip"
OUT = Path("docs/measurements/source-copy-installation")
DONOR = Path("exports/conditional-reentry-backbone") / PACKED
BASE = Path("exports/tool-language-packed-05")
CHECKS = OUT / "archive.json"


def assemble():
    fit = RESULT / FIT
    identity = read(fit / "fit.json")
    assert (
        digest(DONOR) == identity["backbone"]["sha256"]
        and digest(BASE / "tool-export-files.json") == CERTIFIED_INVENTORY
    )
    index = read(BASE / "tool-export-files.json")
    rev = subprocess.check_output(["git", "rev-parse", REV], text=True).strip()
    docs = unpack(Path("research/sources/content-read/training.json.gz"))
    if isinstance(docs, dict):
        docs = docs["rows"]
    finance = min(
        (w for w in docs if w["group"] == "calculate"),
        key=lambda w: fingerprint([87731, w["id"]]),
    )
    reading = min(
        read(READING)["episodes"]["train"], key=lambda w: fingerprint([87731, w["id"]])
    )
    body = next(
        r["story"]
        for r in read(CACHE / "coqa-train-v1.0.json")["data"]
        if r["id"] == reading["id"]
    )
    texts = {
        "report": finance["messages"][0]["content"],
        "passage": body + "\n\n" + reading["turns"][0]["question"],
        "follow-up": reading["turns"][1]["question"],
    }
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        wheel = build(work / "one", rev)
        assert digest(wheel) == digest(build(work / "two", rev))
        guide = work / "README.md"
        guide.write_bytes(subprocess.check_output(
            ["git", "show", f"{GUIDE_REV}:docs/SOURCE_COPY_MODEL.md"], cwd=ROOT
        ))
        files = {
            "README.md": guide,
            "runtime/" + wheel.name: wheel,
            "runtime/uv.lock": ROOT / "uv.lock",
            "runtime/pyproject.toml": ROOT / "pyproject.toml",
            "runtime/LICENSE": ROOT / "LICENSE",
            "conditional-reentry-backbone/" + PACKED: DONOR,
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
            *(k for k in index if k.startswith("chat_templates/")),
        ):
            if name in index:
                assert digest(BASE / name) == index[name]
                files["checkpoint/" + name] = BASE / name
        for path in fit.iterdir():
            if path.is_file():
                files["policy/" + path.name] = path
        for k, v in texts.items():
            f = work / (k + ".txt")
            f.write_text(v)
            files["examples/" + f.name] = f
        rec = save_archive(MODEL, files)
    persist(
        CHECKS,
        dict(
            archive=rec,
            code=rev,
            weights=identity["weights"],
            donor=identity["backbone"]["sha256"],
            inputs=[finance["id"], reading["id"]],
        ),
    )
    print("source-copy archive bytes", rec["bytes"], flush=True)


def verify():
    proof = read(CHECKS)
    arc = proof["archive"]
    assert digest(MODEL) == arc["sha256"]
    verify_members(MODEL, arc["members"], "SHA256SUMS")
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        with ZipFile(MODEL) as z:
            z.extractall(work)
        (wheel,) = (work / "runtime").glob("*.whl")
        site = work / "installed"
        with ZipFile(wheel) as z:
            z.extractall(site)
        env = dict(
            os.environ,
            HF_HOME=str(work / "empty-cache"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        out = []
        for name, flags in (
            ("report", ["--calculator"]),
            ("passage", ["--follow-up", (work / "examples/follow-up.txt").read_text()]),
        ):
            results = []
            for code, checkpoint, policy in (
                (ROOT, ROOT / BASE, ROOT / RESULT / FIT),
                (site, work / "checkpoint", work / "policy"),
            ):
                cmd = [
                    sys.executable,
                    "-I",
                    "-c",
                    NETWORK_GUARD + CODE,
                    str(code),
                    "ask",
                    "--checkpoint",
                    str(checkpoint),
                    "--session-policy",
                    str(policy),
                    "--tool-format",
                    "qwen",
                    "--trace",
                    "--max-tokens",
                    "192",
                    "--max-turns",
                    "3",
                    "--prompt-file",
                    str(work / "examples" / (name + ".txt")),
                    *flags,
                ]
                response = json.loads(subprocess.check_output(cmd, cwd=work, env=env))
                results.append(
                    response["runs"]["recurrent"] if "runs" in response else response
                )

            def rows(x):
                if isinstance(x, list):
                    return x
                if "turns" in x:
                    return [x]
                return x["exchanges"]

            reference = rows(results[0])
            loaded = rows(results[1])
            assert len(reference) == len(loaded) == (2 if name == "passage" else 1)
            for a, b in zip(reference, loaded, strict=True):
                assert trace(a) == trace(b)
                out.append(
                    dict(
                        example=name,
                        trace=fingerprint(trace(a)),
                        tokens=sum(len(t["token_ids"]) for t in a["turns"]),
                        callbacks=sum("observation" in t for t in a["turns"]),
                        status=a["status"],
                        answer=a["answer"],
                    )
                )
    persist(
        OUT / "offline.json",
        dict(
            passed=True,
            archive=arc["sha256"],
            requests=out,
            protocol="Separate workspace and installed CLI processes, network disabled, empty Hugging Face cache, shared preinstalled native dependencies. Compare every native prompt ID, generated ID, actual callback, terminal status and physical attention history, including the follow-up after each model's own first reply.",
        ),
    )
    print([(r["example"], r["tokens"], r["status"]) for r in out])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("build", "verify"))
    assemble() if p.parse_args().action == "build" else verify()
