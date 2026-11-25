"""Run the registered full-history prompt through the deployment generator."""

import gc
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import REVISION, checkpoint
from ..evaluation.answers import numeric
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base, interpret
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..research import core

REG = Path("research/experiments/context-128k-01.json")


def ledger_prompt(t, spec):
    rng = random.Random(spec["seed"])
    total = spec["prompt_tokens"]
    question = "Read the ledger.\n"
    suffix = "\nWhat is the entry for Juniper? Reason briefly and give its number as Final answer: <number>."
    template = t.apply_chat_template(
        [{"role": "user", "content": question + "BODY" + suffix}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    before, after = template.split("BODY")
    encode = lambda s: t.encode(s, add_special_tokens=False)
    prefix, tail = encode(before), encode(after)
    gold = str(rng.randrange(100000, 1000000))
    value = encode(f"Juniper = {gold}\n")
    fill = []
    while len(fill) < total:
        lines = []
        for _ in range(256):
            code = rng.randrange(100000, 1000000)
            lines.append(f"Record {len(fill)}-{len(lines)} = {code}\n")
        fill.extend(encode("".join(lines)))
    position = total // 10
    ids = prefix + fill[: position - len(prefix)] + value
    ids += fill[position : position + total - len(ids) - len(tail)] + tail
    prompt = t.decode(ids)
    # BPE can merge at joins. Pad the difference using independently tokenized words.
    count = len(encode(prompt))
    if count > total:
        raise ValueError("Ledger tokenizer joins exceeded target")
    if count < total:
        prompt = prompt.replace(suffix, " note" * (total - count) + suffix)
    ids = encode(prompt)
    if (
        len(ids) != total
        or not prompt.endswith("<think>\n")
        or prompt.count("Juniper =") != 1
    ):
        raise ValueError("Ledger prompt identity or length differs")
    return prompt, ids, gold


def parity(m, ids, guard, scratch):
    from ..offloaded_prefill import OffloadedPrefill

    m.eval()
    ids = ids[:259]
    cache = m.make_cache()
    for first in range(0, len(ids), 128):
        guard()
        h = m.hidden(mx.array([ids[first : first + 128]]), cache)
        mx.eval(h)
    reference = m.project(h[:, -1:])
    errors = []
    with TemporaryDirectory(dir=scratch) as folder:
        disk = OffloadedPrefill(m, folder)
        actual = disk.prefill(ids, lambda _: guard())
        for i in range(4):
            guard()
            errors.append(
                mx.max(
                    mx.abs(reference.astype(mx.float32) - actual.astype(mx.float32))
                ).item()
            )
            if errors[-1] != 0:
                break
            if i < 3:
                token = mx.argmax(reference[:, -1], -1)[:, None]
                reference = m.project(m.hidden(token, cache))
                actual = disk.decode(token)
    return errors


def run(offload=False, recovery=False):
    reg = REG.with_name("context-128k-disk.json") if offload else REG
    spec = read(reg)
    out = (core.OUT / "recovered" if recovery else core.OUT) / spec["id"]
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    t = load_tokenizer(local_snapshot())
    prompt, ids, gold = ledger_prompt(t, spec)
    packed(
        out / "prompt.json.gz",
        {"prompt_ids": ids, "gold": gold, "input_sha256": fingerprint(ids)},
    )
    if offload:
        from ..evaluation.mechanism import unpack

        original = unpack(
            Path("research/results") / spec["reference"] / "prompt.json.gz"
        )
        if original["prompt_ids"] != ids or original["gold"] != gold:
            raise ValueError("Frozen long-context query changed")
    candidate = Path(spec["candidate"])
    if digest(candidate / "adapters.safetensors") != spec["weights"]:
        raise ValueError("Wrong recurrent checkpoint")
    core.assets()
    b = core.Budget(spec["id"], spec["limits"])
    rows = {}
    progress = 0
    try:
        if offload:
            errors = {}
            for arm in ("base", "candidate"):
                start = time.perf_counter()
                m, t, _ = (
                    base() if arm == "base" else checkpoint.load_trained(candidate)
                )
                errors[arm] = parity(m, ids, b.check, out)
                save(out / "parity.json", errors)
                del m, t
                gc.collect()
                mx.clear_cache()
                if errors[arm] != [0.0] * 4:
                    save(
                        out / "decision.json",
                        {
                            "status": "parity_failed",
                            "capacity": False,
                            "completed": [],
                            "base_revision": REVISION,
                            "registration": digest(reg),
                            "arm": arm,
                            "errors": errors[arm],
                        },
                    )
                    return
        for arm in ("base", "candidate"):
            progress = 0
            start = time.perf_counter()
            if arm == "base":
                m, t, weight = base()
            else:
                m, t, _ = checkpoint.load_trained(candidate)
                weight = spec["weights"]
            if m.span.passes != (1 if arm == "base" else 2):
                raise ValueError("Unexpected depth")
            if t.encode(prompt, add_special_tokens=False) != ids:
                raise ValueError("Tokenizer changed")

            def check(position, arm=arm, start=start):
                nonlocal progress
                advanced = position != progress
                progress = position
                b.check()
                if advanced and position % 8192 == 0:
                    print(
                        arm, position, round(time.perf_counter() - start, 2), flush=True
                    )

            end = core.atom(t, "<|im_end|>")
            trial = generate(
                m,
                t,
                prompt,
                spec["generation_cap"],
                progress=check,
                stop_ids=[end],
                offload_dir=out if offload else None,
            )
            trial.pop("prompt")
            stop = set(t.eos_token_ids) | {end}
            parsed = interpret(
                trial["token_ids"],
                core.atom(t, "<think>"),
                core.atom(t, "</think>"),
                stop,
                t,
            )
            answer = (
                numeric(parsed["visible"]) if trial["token_ids"][-1] in stop else None
            )
            rows[arm] = dict(
                **trial,
                weights=weight,
                input_sha256=fingerprint(ids),
                answer=answer,
                correct=answer == gold,
            )
            save(out / f"{arm}.json", rows[arm])
            del m
            gc.collect()
            mx.clear_cache()
        result = {
            "status": "completed",
            "capacity": True,
            "retrieval": {a: r["correct"] for a, r in rows.items()},
        }
        if offload and recovery:
            resident = Path("research/sessions/2026-11-25") / spec["reference"]
            result["resident_archive"] = str(resident)
            result["identical_generations"] = {}
            for arm in rows:
                baseline = read(resident / f"{arm}.json")
                if (baseline["weights"], baseline["input_sha256"]) != (
                    rows[arm]["weights"],
                    rows[arm]["input_sha256"],
                ):
                    raise ValueError("Resident comparison differs")
                result["identical_generations"][arm] = (
                    rows[arm]["token_ids"] == baseline["token_ids"]
                )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result = {
            "status": "interrupted",
            "arm": arm,
            "processed_positions": progress,
            "reason": str(exc),
            "arm_wall_seconds": time.perf_counter() - start,
            "peak_bytes": mx.get_peak_memory(),
            "capacity": False,
        }
    finally:
        b.close()
    save(
        out / "decision.json",
        dict(
            **result,
            base_revision=REVISION,
            registration=digest(reg),
            completed=list(rows),
        ),
    )
    print(result, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    run(args.offload, args.recovery)
