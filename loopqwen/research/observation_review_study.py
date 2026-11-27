"""Preregistered free exchanges and matched post-execution decisions on native MLX."""

import argparse
import gc
import math
import random
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from .. import package_tools, qwen_tools
from ..calculator_tool import tool
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..observation_review import ObservationAssistant, backbone
from ..tool_compare import DenseAssistant
from ..tool_model import tool_view
from ..tool_session import compact
from .core import Budget
from .native_function_data import catalog, token_pair
from .native_function_study import (
    evaluate,
    executions,
    fit_messages,
    grade,
    replay,
    summarize,
)
from .registered import registered

FAMILIES = ("ASDiv", "SVAMP", "public", "packages", "files")


def choices(tok, row, kind):
    calc = tool()
    call = row[kind]
    value = calc.callback(**call["arguments"])
    assert (Fraction(value["value"]) == Fraction(row["reference"])) == (kind == "right")
    item = {
        "tools": qwen_tools.declarations([calc]),
        "messages": [
            {"role": "user", "content": row["question"]},
            qwen_tools.assistant_message([call]),
            {"role": "tool", "content": compact(value)},
        ],
    }
    prefix, accept = token_pair(
        tok, dict(item, target={"role": "assistant", "content": value["value"]})
    )
    other, retry = token_pair(
        tok, dict(item, target=qwen_tools.assistant_message([row["right"]]))
    )
    assert prefix == other
    index = next(i for i, (a, b) in enumerate(zip(accept, retry)) if a != b)
    return prefix + accept[:index], [accept[index], retry[index]]


def probe(net, tok, panel, budget):
    rows = []
    for row in panel["probe"]:
        record = {"id": row["id"]}
        for kind in ("right", "wrong"):
            budget.check()
            ids, alternatives = choices(tok, row, kind)
            scores = (
                net.project(net.hidden(mx.array([ids]))[:, -1:])[
                    0, 0, mx.array(alternatives)
                ]
                .astype(mx.float32)
                .tolist()
            )
            record[kind] = {
                "prefix": fingerprint(ids),
                "length": len(ids),
                "tokens": alternatives,
                "logits": scores,
            }
        rows.append(record)
    return rows


def passed_probe(row, kind):
    a, b = row[kind]["logits"]
    return a > b if kind == "right" else a < b


def paired(a, b):
    wins, losses = len(a - b), len(b - a)
    return {"wins": wins, "losses": losses, "p": exact_mcnemar(wins, losses)}


def outcome(panel, data, probes):
    groups = {}
    free = {}
    answers = {}
    mechanism = {}
    review = {}
    for side, arm in data.items():
        groups[side] = {
            f: summarize([r for r in arm if f == "all" or f == r["family"]])
            for f in ("all", *FAMILIES)
        }
        free[side] = {
            r["id"] for r in arm if all(q["score"]["success"] for q in r["requests"])
        }
        answers[side] = set()
        counts = {"initial_wrong": 0, "repaired": 0, "initial_right": 0, "kept": 0}
        for w, r in zip(panel["rows"], arm, strict=True):
            if w["family"] not in FAMILIES[:2]:
                continue
            q = r["requests"][0]
            if q["score"]["correct_answer"]:
                answers[side].add(w["id"])
            calls = executions(q["trajectory"])
            if not calls or "value" not in calls[0]["observation"]:
                continue
            correct = Fraction(calls[0]["observation"]["value"]) == Fraction(w["gold"])
            counts["initial_right" if correct else "initial_wrong"] += 1
            counts["kept" if correct else "repaired"] += int(q["score"]["success"])
        review[side] = counts
        mechanism[side] = {
            r["id"]
            for r in probes[side]
            if all(passed_probe(r, k) for k in ("right", "wrong"))
        }
    paired_free, paired_answers, paired_probe = (
        paired(z["candidate"], z["control"]) for z in (free, answers, mechanism)
    )
    c, b = groups["candidate"], groups["control"]
    gates = dict(
        fresh=paired_answers["wins"] - paired_answers["losses"] >= 4
        and paired_answers["p"] <= 0.05,
        each_source=all(
            c[f][m] >= b[f][m] for f in FAMILIES[:2] for m in ("answers", "grounded")
        ),
        repairs=review["candidate"]["repaired"] >= 4,
        retention=all(c[f]["strict"] >= b[f]["strict"] for f in FAMILIES[2:]),
        strict=paired_free["wins"] > paired_free["losses"] and paired_free["p"] <= 0.05,
        mechanism=len(mechanism["candidate"]) >= 16
        and paired_probe["wins"] > paired_probe["losses"]
        and paired_probe["p"] <= 0.05,
        **{m: c["all"][m] <= b["all"][m] for m in ("wall_s", "first_s", "peak")},
    )
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": groups,
        "strict": paired_free,
        "human": paired_answers,
        "receipts": dict(
            paired=paired_probe,
            **{
                s: dict(
                    pairs=len(mechanism[s]),
                    **{
                        k: sum(passed_probe(r, k) for r in rows)
                        for k in ("right", "wrong")
                    },
                )
                for s, rows in probes.items()
            },
        ),
        "free_receipts": review,
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    panel = read(spec["material"])
    train = read(spec["training"])
    tok = load_tokenizer(local_snapshot())
    meta = read(root / "fit.json")
    assert all(digest(Path(k)) == v for k, v in train["sources"].items())
    assert meta["weights"] == digest(root / "tool-adapters.safetensors") and meta[
        "training"
    ] == digest(Path(spec["training"]))
    assert (meta["trainable"], meta["passes"], meta["updates"]) == (
        802816,
        2,
        spec["updates"],
    )
    lengths = [len(token_pair(tok, x)[1]) for x in train["examples"]]
    rng = random.Random(spec["fit_seed"])
    assert (
        len(lengths) == meta["turns"]
        and sum(sum(rng.sample(lengths, spec["batch"])) for _ in range(spec["updates"]))
        == meta["targets"]
    )
    held = {w["id"] for w in panel["rows"] + panel["probe"]}
    assert len(held) == 106 and all(w["id"] not in held for w in train["paired_math"])
    calc = tool()
    for w in train["paired_math"]:
        for k in ("right", "wrong"):
            v = calc.callback(**w[k]["arguments"])["value"]
            assert (Fraction(v) == Fraction(w["reference"])) == (k == "right")
    data, probes = {}, {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        probes[side] = read(root / (side + "-probe.json"))
        for w, r in zip(panel["probe"], probes[side], strict=True):
            assert w["id"] == r["id"]
            for k in ("right", "wrong"):
                ids, alts = choices(tok, w, k)
                assert (fingerprint(ids), len(ids), alts) == (
                    r[k]["prefix"],
                    r[k]["length"],
                    r[k]["tokens"],
                )
                assert len(r[k]["logits"]) == 2 and all(
                    map(math.isfinite, r[k]["logits"])
                )
        for w, r in zip(panel["rows"], data[side], strict=True):
            assert (w["id"], w["family"]) == (r["id"], r["family"])
            assert (
                r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                == w["initial_ids"]
            )
            history = []
            tools = catalog(w, panel)
            for i, (question, q) in enumerate(
                zip(w["questions"], r["requests"], strict=True)
            ):
                history.append({"role": "user", "content": question})
                replay(tok, w, tools, history, q, spec, side, after_callback=True)
                assert grade(w, i, q["trajectory"]) == q["score"]
    assert outcome(panel, data, probes) == decision


def optimize(spec, root, budget):
    text = backbone(spec["checkpoint"], spec)
    tok = load_tokenizer(local_snapshot())
    mx.random.seed(spec["fit_seed"])
    net = tool_view(text, spec["start_layer"], spec["rank"], 2)
    fit_messages(net, tok, spec, root, budget)


def run(name):
    spec = registered(name)
    release = Path(spec["release"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    directory = Path(spec["checkpoint"])
    assert read(release / "export-files.json") == read(
        directory / "tool-export-files.json"
    )
    assert all(
        digest(directory / k) == v
        for k, v in read(release / "export-files.json").items()
    )
    panel = read(spec["material"])
    assert package_tools.snapshot() == panel["environment"] and all(
        digest(Path(k)) == v["sha256"] for k, v in panel["files"].items()
    )
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            optimize(spec, root, budget)

        def factory(side):
            return (
                ObservationAssistant(directory, root)
                if side == "candidate"
                else DenseAssistant(True, "qwen")
            )

        data, probes = {}, {}
        for side in ("candidate", "control"):
            gc.collect()
            mx.clear_cache()
            dest = root / (side + ".json.gz")
            if not dest.exists():
                packed(
                    dest,
                    evaluate(
                        spec, panel, side, budget, lambda s=side: factory(s), catalog
                    ),
                )
            data[side] = unpack(dest)
        for side in ("candidate", "control"):
            gc.collect()
            mx.clear_cache()
            actor = factory(side)
            if side == "candidate":
                actor.load_tools()
                net = actor.review
            else:
                net = actor.language
            probes[side] = probe(net, actor.tokenizer, panel, budget)
            save(root / (side + "-probe.json"), probes[side])
            del actor, net
        save(root / "decision.json", outcome(panel, data, probes))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
