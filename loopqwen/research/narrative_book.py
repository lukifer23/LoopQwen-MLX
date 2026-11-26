"""Read two full-length versions of one natural book through the shipped assistant's language route."""

import argparse
import gc
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base, interpret
from ..evaluation.mechanism import packed, unpack
from ..export_runtime import load_export
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from .core import Budget
from .registered import registered


def grade(tok, ids, aliases, others):
    term = {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
    parsed = interpret(
        ids,
        tok.convert_tokens_to_ids("<think>"),
        tok.convert_tokens_to_ids("</think>"),
        term,
        tok,
    )
    visible = parsed["visible"] if ids[-1] in term else ""
    norm = " ".join(re.findall(r"\w+", visible.casefold()))

    def contains(s):
        return " " + s.lower() + " " in " " + norm + " "

    return dict(
        visible=visible,
        exact=norm in [s.casefold() for s in aliases],
        content=any(map(contains, aliases)) and not any(map(contains, others)),
    )


def outcome(data, material):
    metrics = {}
    proof = True
    for side, rs in data.items():
        metrics[side] = dict(
            correct=sum(r["grade"]["exact"] for r in rs),
            content=sum(r["grade"]["content"] for r in rs),
            seconds=sum(r["elapsed_s"] for r in rs),
            ttft=sum(r["ttft_s"] for r in rs),
            peak=max(r["peak_bytes"] for r in rs),
            routes=[r.get("context_route", "base") for r in rs],
        )
        for r, w in zip(rs, material["panel"], strict=True):
            n = len(w["prompt_ids"]) + len(r["token_ids"]) - 1
            expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
            if r.get("context_route") in ("register", "short_arithmetic", "entailment"):
                expected.update({(11, 1), (15, 1)})
            proof &= (
                r["input_sha256"] == w["sha256"] == fingerprint(w["prompt_ids"])
                and r["prompt_tokens"] == len(w["prompt_ids"]) >= 131072
                and {(s["layer"], s["turn"]) for s in r["kv_history"]} == expected
                and all(s["offset"] == s["retained"] == n for s in r["kv_history"])
            )
    a, b = metrics["candidate"], metrics["control"]
    return dict(
        passed=bool(
            proof
            and a["correct"] == 2
            and b["correct"] < 2
            and a["seconds"] <= b["seconds"]
            and a["peak"] <= b["peak"]
        ),
        full_history=proof,
        metrics=metrics,
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    material = read(spec["material"])
    tok = load_tokenizer(local_snapshot())
    data = {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for i, (r, w) in enumerate(zip(data[side], material["panel"], strict=True)):
            assert r["id"] == w["id"] and r["grade"] == grade(
                tok,
                r["token_ids"],
                spec["accepted_names"][i],
                spec["accepted_names"][1 - i],
            )
            assert (
                len(r["token_ids"]) <= spec["max_tokens"]
                and 0 < r["ttft_s"] <= r["elapsed_s"]
            )
    assert outcome(data, material) == decision


def run(name):
    spec = registered(name)
    material = read(spec["material"])
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    parent = Path(spec["requires"])
    bundle = Path(spec["checkpoint"])
    if (
        digest(parent / "checksums.json") != spec["prerequisite_sha256"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Unqualified release")
    for file, h in read(parent / "export-files.json").items():
        if digest(bundle / file) != h:
            raise ValueError("Different deployed file: " + file)
    budget = Budget(name, spec["limits"])
    try:
        from .paired_progress import resume

        data = resume(out, spec, material["panel"], ("candidate", "control"))
        for i, w in enumerate(material["panel"]):
            for side in (
                ("candidate", "control") if i == 0 else ("control", "candidate")
            ):
                if len(data[side]) > i:
                    continue
                model, tok = load_export(bundle) if side == "candidate" else base()[:2]
                ids = w["prompt_ids"]
                text = tok.decode(ids)
                assert tok.encode(text, add_special_tokens=False) == ids
                mx.reset_peak_memory()
                res = generate(
                    model,
                    tok,
                    text,
                    spec["max_tokens"],
                    stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                    progress=lambda _: budget.check(),
                    record_kv_history=True,
                )
                del res["prompt"]
                res.update(
                    id=w["id"],
                    input_sha256=fingerprint(ids),
                    grade=grade(
                        tok,
                        res["token_ids"],
                        spec["accepted_names"][i],
                        spec["accepted_names"][1 - i],
                    ),
                )
                data[side].append(res)
                packed(out / (side + ".json.gz"), data[side])
                print(
                    side,
                    w["id"],
                    res.get("context_route"),
                    res["grade"],
                    res["ttft_s"],
                    flush=True,
                )
                del model, tok
                gc.collect()
                mx.clear_cache()
        save(out / "decision.json", outcome(data, material))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
