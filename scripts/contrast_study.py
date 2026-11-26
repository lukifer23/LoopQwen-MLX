"""Paired native reasoning and matched-prefix readout on publisher-authored question edits."""

import argparse
import gc
import re
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from loopqwen.evaluation.generation_eval import generate
from loopqwen.evaluation.language7 import base
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.evaluation.statistics import exact_mcnemar
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.research.tool_fit import fitted
from loopqwen.tool_session import prompt


def label(tok, ids):
    if ids[-1] not in {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}:
        return None
    parts = tok.decode(ids[:-1]).split("</think>", 1)
    if len(parts) != 2:
        return None
    # Accept a leading yes/no in an explanation or a standalone final yes/no.
    text = parts[1].strip()
    matches = re.findall(r"(?im)^\s*(?:answer\s*:\s*)?(yes|no)\b", text)
    return matches[-1].lower() if matches else None


def probe(model, tok, ids):
    h = model.hidden(mx.array([ids]))[:, -1:]
    choices = [tok.encode(v, add_special_tokens=False) for v in ("yes", "no")]
    assert all(len(c) == 1 for c in choices)
    vals = model.project(h)[0, -1, mx.array([c[0] for c in choices])].tolist()
    return vals


def summarize(data, items, spec):
    arms = {}
    wins = losses = 0
    keys = list(dict.fromkeys(w["pair"] for w in items))
    by_arm = {}
    for side, rows in data.items():
        by_pair = {k: True for k in keys}
        for row, w in zip(rows, items, strict=True):
            by_pair[w["pair"]] &= row["correct"]
        by_arm[side] = by_pair
        arms[side] = dict(
            correct=sum(r["correct"] for r in rows),
            pairs=sum(by_pair.values()),
            probe=sum(r["probe_correct"] for r in rows),
            seconds=sum(r["seconds"] for r in rows),
            probe_seconds=sum(r["probe_seconds"] for r in rows),
            tokens=sum(len(r["ids"]) for r in rows),
            completed=sum(r["label"] is not None for r in rows),
        )
    for k in keys:
        a, b = (by_arm[s][k] for s in ("candidate", "control"))
        wins += a and not b
        losses += b and not a
    p = exact_mcnemar(wins, losses)
    a, b = arms.values()
    return dict(
        passed=(
            wins - losses >= spec["pair_floor"]
            and p <= 0.05
            and a["probe"] - b["probe"] >= spec["probe_floor"]
            and a["seconds"] <= spec["time_ratio"] * b["seconds"]
        ),
        metrics=arms,
        pair_gains=wins,
        pair_losses=losses,
        p_exact=p,
        pairs=len(keys),
        items=len(items),
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    items = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    rows = {}
    for side in ("candidate", "control"):
        rows[side] = unpack(root / (side + ".json.gz"))
        for r, w in zip(rows[side], items, strict=True):
            text = prompt(tok, w["messages"])
            labs = label(tok, r["ids"])
            pred = ("yes", "no")[r["scores"][1] > r["scores"][0]]
            assert r["id"] == w["id"] and r["prompt_ids"] == tok.encode(
                text, add_special_tokens=False
            )
            assert r["probe_ids"] == tok.encode(
                text + "</think>\n\n", add_special_tokens=False
            )
            assert (
                labs == r["label"]
                and r["correct"] == (labs == w["gold"])
                and r["probe_correct"] == (pred == w["gold"])
            )
            assert (
                len(r["ids"]) <= spec["max_tokens"]
                and r["seconds"] > 0
                and r["probe_seconds"] > 0
            )
    assert decision == summarize(rows, items, spec)
    if (root / "fit.json").exists():
        meta = read(root / "fit.json")
        assert meta["training"] == digest(Path(spec["training"])) and meta[
            "weights"
        ] == digest(root / "tool-adapters.safetensors")


def main(name):
    spec = registered(name)
    items = read(spec["material"])["rows"]
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise ValueError("Existing run")
    if "requires" in spec:
        pre = Path(spec["requires"])
        if (
            digest(pre / "checksums.json") != spec["prerequisite_sha256"]
            or not read(pre / "decision.json")["passed"]
        ):
            raise ValueError("Reasoning prerequisite failed")
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(name, spec["limits"])
    rows = {}
    try:
        for side in ("candidate", "control"):
            model, tok = (
                base()[:2]
                if spec[side] == "base"
                else fitted(spec[side], spec, out, budget)
            )
            samples = []
            for w in items:
                budget.check()
                text = prompt(tok, w["messages"])
                ids = tok.encode(text, add_special_tokens=False)
                common = tok.encode(text + "</think>\n\n", add_special_tokens=False)
                at = perf_counter()
                logits = probe(model, tok, common)
                probe_time = perf_counter() - at
                val = generate(
                    model,
                    tok,
                    text,
                    spec["max_tokens"],
                    stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                    progress=lambda _: budget.check(),
                )
                ans = label(tok, val["token_ids"])
                prediction = ("yes", "no")[logits[1] > logits[0]]
                samples.append(
                    dict(
                        id=w["id"],
                        label=ans,
                        correct=ans == w["gold"],
                        scores=logits,
                        probe_correct=prediction == w["gold"],
                        prompt_ids=ids,
                        probe_ids=common,
                        ids=val["token_ids"],
                        seconds=val["elapsed_s"],
                        probe_seconds=probe_time,
                    )
                )
                if len(samples) % 8 == 0:
                    print(
                        side,
                        len(samples),
                        sum(r["correct"] for r in samples),
                        flush=True,
                    )
            packed(out / (side + ".json.gz"), samples)
            rows[side] = samples
            del model, tok
            gc.collect()
            mx.clear_cache()
        assert all(
            a["prompt_ids"] == b["prompt_ids"] and a["probe_ids"] == b["probe_ids"]
            for a, b in zip(*rows.values(), strict=True)
        )
        d = summarize(rows, items, spec)
        save(out / "decision.json", d)
        print(d, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    main(parser.parse_args().name)
