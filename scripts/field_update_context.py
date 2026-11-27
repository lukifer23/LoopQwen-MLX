"""Prepare and verify the qualified release's original full-context CLI requests."""

import json
from time import perf_counter

from mlx_lm.utils import load_tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt


def prepare(root, files):
    release = root / "research/results/packed-release-retention-04"
    cert = read(release / "registration.json")
    source = root / cert["register_material"]
    assert digest(source) == cert["register_material_sha256"]
    tok = load_tokenizer(local_snapshot())
    before, after = prompt(tok, [{"role": "user", "content": "INPUT_MARKER"}]).split(
        "INPUT_MARKER"
    )
    expected = []
    for i, row in enumerate(read(source)[0]["panel"]):
        name = f"examples/register-{i}.txt"
        ids = row["prompt_ids"]
        rendered = tok.decode(ids)
        assert rendered.startswith(before) and rendered.endswith(after)
        body = rendered[len(before) : -len(after)]
        assert (
            tok.encode(
                prompt(tok, [{"role": "user", "content": body}]),
                add_special_tokens=False,
            )
            == ids
        )
        files[name] = body.encode()
        reference = release / f"candidate-{i}.json"
        assert digest(reference) == read(release / "checksums.json")[reference.name]
        archived = read(reference)["register"]
        assert archived["input_sha256"] == fingerprint(ids) and len(ids) >= 131072
        evidence = {
            "source": reference.relative_to(root).as_posix(),
            "source_sha256": digest(reference),
            "gold": row["gold"],
            **{
                k: archived[k]
                for k in (
                    "input_sha256",
                    "token_ids",
                    "kv_history",
                    "page_sites",
                    "context_route",
                    "prefill_policy",
                    "prompt_tokens",
                )
            },
        }
        ref = f"examples/register-{i}-reference.json"
        files[ref] = (json.dumps(evidence, indent=2) + "\n").encode()
        expected.append((name, ref))
    assert len(expected) == 2
    return expected


def verify(work, examples, execute, receipt):
    results = []
    for name, ref in examples:
        expected = read(work / ref)
        args = [
            "ask",
            "--checkpoint",
            "release-05",
            "--prompt-file",
            name,
            "--max-tokens",
            "96",
            "--trace",
        ]
        start = perf_counter()
        actual = execute(args)
        wall = perf_counter() - start
        (turn,) = actual["turns"]
        assert (
            actual["status"] == "answered"
            and actual["answer"].removeprefix("Final answer:").strip()
            == expected["gold"]
        )
        assert (actual["context_route"], actual["prefill_policy"]) == (
            expected["context_route"],
            expected["prefill_policy"],
        )
        assert fingerprint(turn["prompt_ids"]) == expected["input_sha256"]
        assert all(
            turn[k] == expected[k] for k in ("token_ids", "kv_history", "page_sites")
        )
        assert len(turn["prompt_ids"]) == expected["prompt_tokens"] >= 131072
        results.append(
            {
                "input": name,
                "answer": actual["answer"],
                "actual_initial_tokens": len(turn["prompt_ids"]),
                "input_sha256": expected["input_sha256"],
                "token_ids": turn["token_ids"],
                "physical_histories": turn["kv_history"],
                "page_site_counters_identical": True,
                "peak_bytes": turn["peak_bytes"],
                "native_first_seconds": turn["ttft_s"],
                "native_seconds": turn["seconds"],
                "process_wall_seconds": wall,
            }
        )

    def stable(row):
        return {
            k: v
            for k, v in row.items()
            if k
            not in (
                "peak_bytes",
                "native_first_seconds",
                "native_seconds",
                "process_wall_seconds",
            )
        }

    if receipt.exists():
        assert [stable(r) for r in read(receipt)["requests"]] == [
            stable(r) for r in results
        ]
    else:
        save(
            receipt,
            {
                "tested_policy": "original qualified release 05 default, installed update wheel",
                "requests": results,
            },
        )
    return [stable(r) for r in results]
