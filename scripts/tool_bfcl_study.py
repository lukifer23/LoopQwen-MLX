"""Evaluate tool selection and typed arguments on unmodified publisher math questions."""

import argparse
import gc
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from jsonschema import Draft202012Validator
from mlx_lm.utils import load_tokenizer

from loopqwen.evaluation.generation_eval import generate
from loopqwen.evaluation.language7 import base
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.evaluation.statistics import exact_mcnemar
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.research.tool_fit import restore
from loopqwen.tool_session import SYSTEM, compact, decode_action, prompt


def normalized_definition(definition):
    f = deepcopy(definition)

    def normalize(node):
        if "type" in node:
            node["type"] = {
                "dict": "object",
                "float": "number",
                "tuple": "array",
                "int": "integer",
                "bool": "boolean",
                "str": "string",
                "list": "array",
            }.get(node["type"], node["type"])
        for v in node.get("properties", {}).values():
            normalize(v)
        if "items" in node:
            normalize(node["items"])

    normalize(f["parameters"])
    return f


def render(tok, item):
    return prompt(
        tok,
        [
            dict(
                role="system",
                content=SYSTEM
                + compact([normalized_definition(f) for f in item["function"]])
                + "\nUse the supplied function for the user's request.",
            ),
            item["question"][0][0],
        ],
    )


def judgment(item, tokens, tok):
    if tokens[-1] not in {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}:
        return dict(complete=False, valid=False, correct=False)
    text = tok.decode(tokens[:-1])
    if "</think>" not in text:
        return dict(complete=False, valid=False, correct=False)
    try:
        obj = decode_action(text.split("</think>", 1)[1].strip())
    except ValueError:
        obj = None
    valid = False
    correct = False
    definitions = {f["name"]: normalized_definition(f) for f in item["function"]}
    if obj and obj["name"] in definitions:
        args = obj["arguments"]
        valid = Draft202012Validator(definitions[obj["name"]]["parameters"]).is_valid(
            args
        )
        for option in item["ground_truth"]:
            truth = option.get(obj["name"])
            if truth is None:
                continue
            if (
                valid
                and not (args.keys() - truth.keys())
                and all(
                    (k in args and args[k] in vals) or (k not in args and "" in vals)
                    for k, vals in truth.items()
                )
            ):
                correct = True
                break
    return dict(complete=True, valid=bool(valid), correct=bool(correct))


def summary(rows, spec):
    scores = {
        a: {
            k: sum(r[k] for r in items)
            for k in ("complete", "valid", "correct", "seconds")
        }
        for a, items in rows.items()
    }
    pairs = list(zip(rows["candidate"], rows["control"], strict=True))
    wins = sum(a["correct"] and not b["correct"] for a, b in pairs)
    losses = sum(b["correct"] and not a["correct"] for a, b in pairs)
    p = exact_mcnemar(wins, losses)
    passed = (
        wins - losses >= spec["net_floor"]
        and p <= spec["p_max"]
        and scores["candidate"]["seconds"]
        <= scores["control"]["seconds"] * spec["time_ratio"]
    )
    extra = {}
    if spec.get("retention"):
        items = read(spec["material"])["rows"]
        counts = {}
        for tag in ("simple", "multiple", "fresh"):
            counts[tag] = [
                sum(
                    x["correct"]
                    for x, w in zip(arm, items, strict=True)
                    if w["panel"] == tag
                )
                for arm in rows.values()
            ]
        evidence = {}
        for parent in spec["reference_traces"]:
            for r in unpack(Path(parent) / "candidate.json.gz"):
                evidence[r["id"]] = r
        parity = all(
            r[k] == evidence[r["id"]][k]
            for r in rows["control"]
            if r["id"] in evidence
            for k in ("prompt_ids", "token_ids")
        )
        passed = (
            parity
            and counts["simple"][0] >= 16
            and counts["multiple"][0] >= 18
            and counts["fresh"][0] >= counts["fresh"][1] - 1
            and wins >= losses
            and scores["candidate"]["seconds"]
            <= scores["control"]["seconds"] * spec["time_ratio"]
        )
        extra["retention"] = dict(panels=counts, prior_exact=parity)
    return dict(
        **extra,
        passed=passed,
        metrics=scores,
        gains=wins,
        losses=losses,
        p_exact=p,
        items=len(pairs),
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    items = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    rows = {a: unpack(root / (a + ".json.gz")) for a in ("candidate", "control")}
    for arm in rows.values():
        for r, w in zip(arm, items, strict=True):
            if (
                r["id"] != w["id"]
                or r["prompt_ids"]
                != tok.encode(render(tok, w), add_special_tokens=False)
                or r["seconds"] <= 0
                or not 1 <= len(r["token_ids"]) <= spec["max_tokens"]
                or any(r[k] != v for k, v in judgment(w, r["token_ids"], tok).items())
            ):
                raise ValueError("External argument evidence differs")
    if summary(rows, spec) != decision:
        raise ValueError("External summary differs")


def main(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    src = Path(spec["source"])
    if (
        digest(src / "checksums.json") != spec["archive"]
        or not read(src / "decision.json")["passed"]
    ):
        raise ValueError("External transfer prerequisite failed")
    if (out / "decision.json").exists():
        raise ValueError("Completed run exists")
    items = read(spec["material"])["rows"]
    budget = Budget(name, spec["limits"])
    rows = {}
    try:
        for side in ("candidate", "control"):
            if side == "candidate":
                model, tok = restore(spec["model"])
            elif spec.get("control_model"):
                parent = Path(spec["control_model"])
                if digest(parent / "checksums.json") != spec["control_sha256"]:
                    raise ValueError("Different structural retention control")
                model, tok = restore(parent)
            else:
                model, tok = base()[:2]
            pars = []
            for w in items:
                text = render(tok, w)
                result = generate(
                    model,
                    tok,
                    text,
                    spec["max_tokens"],
                    progress=lambda _: budget.check(),
                    stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                )
                tokens = result["token_ids"]
                pars.append(
                    dict(
                        id=w["id"],
                        prompt_ids=tok.encode(text, add_special_tokens=False),
                        token_ids=tokens,
                        seconds=result["elapsed_s"],
                        ttft_s=result["ttft_s"],
                        **judgment(w, tokens, tok),
                    )
                )
                print(side, w["id"], pars[-1]["correct"], flush=True)
            packed(out / (side + ".json.gz"), pars)
            rows[side] = pars
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(out / "decision.json", summary(rows, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    main(parser.parse_args().name)
