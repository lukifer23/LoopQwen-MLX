"""Export full user inputs and verify genuine installed two-arm, full-history generation traces."""

import argparse
import gzip
from pathlib import Path

from loopqwen import REVISION
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_study import replay
from scripts.locale_context_material import SOURCE, material, request

CAP = 384
ARMS = {"recurrent": "candidate", "base": "control"}


def source_request(w):
    src = gzip.decompress((SOURCE / (w["id"] + ".xml.gz")).read_bytes())
    assert len(src) == w["file_bytes"]
    return request(w["locale"], src.decode("utf-8"))


def export(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for w in material():
        content = source_request(w).encode("utf-8")
        p = folder / (w["id"] + ".txt")
        if p.exists():
            assert p.read_bytes() == content
        else:
            p.write_bytes(content)


def verify(folder, checkpoint, policy):
    # Bind the reported comparison to its independently hashed registered source weights.
    registration = read(policy / "registration.json")
    fit = read(policy / "fit.json")
    weights = policy / (
        "reentry.safetensors"
        if registration.get("reentry_block")
        else "cross-view.safetensors"
    )
    sha = digest(weights)
    inventory = read(policy / "checksums.json")
    assert (
        sha == inventory[weights.name]
        and digest(policy / "fit.json") == inventory["fit.json"]
    )
    assert digest(policy / "registration.json") == inventory["registration.json"]
    if registration.get("reentry_block"):
        assert fit["weights"] == sha and registration["reentry_block"] == [
            20,
            21,
            22,
            23,
        ]
    else:
        assert read(policy / "compiled.json")["weights"] == sha
    tok = tokenizer({"checkpoint": checkpoint})
    ids = unpack(SOURCE / "prompt-ids.json.gz")
    spec = {"max_turns": 1, "max_tokens": CAP}
    rows = []
    for w in material():
        source = source_request(w)
        record = folder / (w["id"] + ".json")
        tr = read(record)
        assert (
            tr["base_revision"] == REVISION
            and tr["base_source"] == "certified_release_text"
        )
        assert (
            tr["order"] == list(ARMS)
            and tr["initial_prompt_identical"]
            and tr["prefill_cache_state"]
        )
        assert tr["max_turns"] == 1 and tr["max_tokens_per_turn"] == CAP
        # Compare only traces produced using the declared absolute model paths. The
        # ordinary CLI validates the files while constructing each model.
        assert tr["tool_format"] == "qwen"
        assert Path(tr["session_policy"]).is_absolute() and Path(
            tr["session_policy"]
        ).resolve(strict=True) == policy.resolve(strict=True)
        assert Path(tr["recurrent_checkpoint"]).is_absolute() and Path(
            tr["recurrent_checkpoint"]
        ).resolve(strict=True) == checkpoint.resolve(strict=True)
        for side, arm in ARMS.items():
            (r,) = tr["runs"][side]
            (turn,) = r["turns"]
            assert r["question"] == source
            assert not turn.get("reused_prompt_tokens", 0)
            assert (
                r["initial_tokens"] == len(ids[w["id"]]) == w["input_tokens"] > 131072
            )
            assert r["initial_sha256"] == fingerprint(ids[w["id"]]) == w["input_sha256"]
            assert turn["prompt_ids"] == ids[w["id"]]
            assert turn["cache_state_blocks"] == (r["initial_tokens"] - 1) // 128
            replay(
                tok,
                w,
                [],
                [{"role": "user", "content": source}],
                {
                    "trajectory": r,
                    "wall_s": r["request_seconds"],
                    "first_s": r["first_token_seconds"],
                    "peak": r["peak_mlx_bytes"],
                },
                spec,
                arm,
            )
            answer = (r["answer"] or "").strip()
            rows.append(
                {
                    "id": w["id"],
                    "locale": w["locale"],
                    "model": side,
                    "expected": w["answer"],
                    "answer": answer,
                    "correct": r["status"] == "answered" and answer == w["answer"],
                    "status": r["status"],
                    "input_tokens": r["initial_tokens"],
                    "output_tokens": len(turn["token_ids"]),
                    "opening_s": turn["ttft_s"],
                    "decode_s": turn["seconds"] - turn["ttft_s"],
                    "wall_s": r["request_seconds"],
                    "first_s": r["first_token_seconds"],
                    "peak": r["peak_mlx_bytes"],
                    "trace_sha256": digest(record),
                }
            )
    metrics = {}
    for arm in ARMS:
        group = [r for r in rows if r["model"] == arm]
        metrics[arm] = {
            "correct": sum(r["correct"] for r in group),
            "whole_locales": sum(
                all(r["correct"] for r in group if r["locale"] == loc)
                for loc in ("en", "it")
            ),
            "wall_s": sum(r["wall_s"] for r in group),
            "first_s": sum(r["first_s"] for r in group),
            "peak_bytes": max(r["peak"] for r in group),
            "opening_tps": sum(r["input_tokens"] for r in group)
            / sum(r["opening_s"] for r in group),
            "decode_tps": sum(r["output_tokens"] - 1 for r in group)
            / sum(r["decode_s"] for r in group),
        }
    return {
        "policy_weights": sha,
        "base_revision": REVISION,
        "metrics": metrics,
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    command = p.add_subparsers(dest="command", required=True)
    make = command.add_parser("inputs")
    make.add_argument("directory", type=Path)
    check = command.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--policy", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if args.command == "inputs":
        export(args.directory)
    else:
        result = verify(args.directory, args.checkpoint, args.policy)
        if args.receipt.is_file():
            assert read(args.receipt) == result
        else:
            save(args.receipt, result)
