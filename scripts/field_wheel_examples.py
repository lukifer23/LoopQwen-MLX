"""Exercise complete document reading and a real two-question exchange from the installed offline wheel."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from document_frequency_material import documents
from portable_base_wheel import NETWORK_GUARD
from verify_runtime_wheel import CODE, ROOT

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save


def main(case):
    receipt = read(ROOT / "docs/measurements/portable-base-wheel-02.json")
    wheel = ROOT / "exports/portable-base-runtime-02" / receipt["wheel"]
    assert digest(wheel) == receipt["sha256"]
    bundle = ROOT / "exports/tool-language-packed-05"
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        site = work / "site"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
        )
        cache = work / "empty-model-cache"
        cache.mkdir()
        cmd = [sys.executable, "-I", "-c", NETWORK_GUARD + CODE, str(site)]
        if case == "document":
            root = ROOT / "research/results/document-frequency-reader-01"
            _, _, source, question, expected, _ = next(documents())
            original = unpack(root / "candidate.json.gz")[0]
            original_base = unpack(root / "control.json.gz")[0]
            request = work / "request.txt"
            request.write_text(source + "\n\n" + question)
            args = [
                "generate",
                "--checkpoint",
                str(bundle),
                "--thinking",
                "--document-reader",
                str(root),
                "--prompt-file",
                str(request),
                "--max-tokens",
                "160",
                "--trace",
                "--compare-base",
            ]
        else:
            root = ROOT / "research/results/question-contrast-01"
            physical = read(
                ROOT / "docs/measurements/observation-review-base-identity.json"
            )
            question = (
                f"A model verifier compared {physical['exact_text_tensors']} original text tensors. "
                f"After quantization the two decoders shared {physical['shared_quantized_and_unquantized_tensors']} arrays "
                f"occupying {physical['shared_bytes']} bytes. How many more shared arrays were there than original text tensors? "
                "Use the calculator for arithmetic."
            )
            args = [
                "ask",
                "--checkpoint",
                str(bundle),
                "--tool-adapter",
                str(root),
                "--tool-format",
                "qwen",
                "--calculator",
                "--prompt",
                question,
                "--follow-up",
                "And how many original text tensors did that verifier compare?",
                "--max-tokens",
                "384",
                "--max-turns",
                "3",
                "--trace",
                "--compare-base",
            ]
        raw = json.loads(
            subprocess.check_output(
                [*cmd, *args],
                cwd=work,
                env=dict(
                    os.environ,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_HOME=str(cache),
                    HF_HUB_CACHE=str(cache / "hub"),
                    HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
                    TRANSFORMERS_CACHE=str(cache / "hub"),
                ),
            )
        )
        assert not any(
            p.is_file()
            and (p.suffix in {".bin", ".safetensors"} or "snapshots" in p.parts)
            for p in cache.rglob("*")
        )
        assert (
            raw["base_source"] == "certified_release_text"
            and raw["initial_prompt_identical"]
        )
    record = {
        "case": case,
        "wheel_sha256": receipt["sha256"],
        "policy": digest(root / "checksums.json"),
        "offline": True,
        "network_connections_blocked": True,
        "empty_model_cache": True,
        "outside_checkout": True,
        "question": question,
    }
    if case == "document":
        candidate, base = (raw["runs"][side] for side in ("recurrent", "base"))
        for side, ref in ((candidate, original), (base, original_base)):
            assert side["initial_sha256"] == ref["input_sha256"]
            for key in ("token_ids", "prompt_tokens", "kv_history"):
                assert side[key] == ref[key]
        for key in ("context_route", "attention_memory"):
            assert candidate[key] == original[key]
        record.update(
            base_source=raw["base_source"],
            initial_prompt_identical=raw["initial_prompt_identical"],
            base={
                k: base[k]
                for k in (
                    "token_ids",
                    "text",
                    "initial_sha256",
                    "kv_history",
                    "cold_first_seconds",
                    "request_seconds",
                    "peak_bytes",
                )
            },
            base_grade=original_base["grade"],
        )
        record.update(
            {
                k: candidate[k]
                for k in (
                    "token_ids",
                    "kv_history",
                    "cold_first_seconds",
                    "request_seconds",
                    "context_route",
                    "attention_memory",
                    "prompt_tokens",
                    "elapsed_s",
                    "ttft_s",
                    "peak_bytes",
                )
            }
        )
        record["rendered_prompt_sha256"] = fingerprint(raw["prompt"])
        record["reference"] = expected
        record["grade"] = original["grade"]
    else:
        record.update(
            result=raw,
            references=[
                physical["shared_quantized_and_unquantized_tensors"]
                - physical["exact_text_tensors"],
                physical["exact_text_tensors"],
            ],
        )
    target = ROOT / f"docs/measurements/{case}-reader-installed.json"
    prior = read(target)
    if case == "document":
        assert prior["token_ids"] == record["token_ids"]
        assert prior["kv_history"] == record["kv_history"]
        assert all(
            prior["base"][k] == record["base"][k] for k in ("token_ids", "kv_history")
        )
    else:
        from verify_runtime_wheel import trace

        for side in ("recurrent", "base"):
            assert [trace(r) for r in raw["runs"][side]] == [
                trace(r) for r in prior["result"]["runs"][side]
            ]
    target = target.with_name(target.stem + "-02.json")
    if target.exists():
        old = read(target)
        for k in ("case", "wheel_sha256", "policy", "question"):
            assert old[k] == record[k]
        if case == "document":
            assert (
                old["base"]["token_ids"] == record["base"]["token_ids"]
                and old["token_ids"] == record["token_ids"]
                and old["rendered_prompt_sha256"] == record["rendered_prompt_sha256"]
            )
        else:
            from verify_runtime_wheel import trace

            for side in ("recurrent", "base"):
                assert [trace(r) for r in raw["runs"][side]] == [
                    trace(r) for r in old["result"]["runs"][side]
                ]
    else:
        save(target, record)
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=("document", "contrast"))
    main(p.parse_args().case)
