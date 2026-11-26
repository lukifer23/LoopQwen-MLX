"""Measure clarification, execution after a follow-up, and retained numerical calls."""

import argparse
import csv
import gc
import io
import re
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..csv_tools import tools as csv_tools
from ..evaluation import executable_tools
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_session import initial_messages
from ..tool_session import run as session
from .core import Budget
from .mixed_field import replay_exchange
from .preference_fit import train
from .raw_distillation import check_weights
from .registered import registered
from .tool_fit import restore


@contextmanager
def documents(material):
    with TemporaryDirectory() as tmp:
        paths = []
        for name, d in material["sheets"].items():
            if Path(name).name != name:
                raise ValueError("Expected a file name")
            path = Path(tmp) / name
            path.write_bytes(d["text"].encode())
            assert digest(path) == d["sha256"]
            paths.append(path)
        yield csv_tools(paths)


def tools_for(w, files):
    if w["kind"] == "public":
        return executable_tools.tools_for(w)
    if w.get("domain") == "zones":
        from ..resource_tools import tools

        return tools("zone")
    return files


def grade(w, xs):
    if w["kind"] == "public":
        g = executable_tools.score(w, xs[0])
        return {
            "success": g["success"],
            "path": g["path"],
            "answer": g["correct_answer"],
            "exact": g["correct_answer"],
            "completed": g["completed"],
            "asked": None,
            "premature": None,
        }
    final = xs[-1]
    calls = [t["call"] for t in final["turns"] if "observation" in t]
    observations = [t["observation"] for t in final["turns"] if "observation" in t]
    path = calls == [w["call"]] and observations == [{"value": w["gold"]}]
    # Permit prose around the answer, requiring a whole value and a completed turn.
    expected = str(w["gold"]).casefold()
    text = (final["answer"] or "").casefold()
    answer = (
        bool(re.search(r"(?<![\w.])" + re.escape(expected) + r"(?!\w|\.\d)", text))
        and final["status"] == "answered"
    )
    scalar = text.strip().strip("'\"`").strip()
    scalar = scalar.removesuffix(".")
    exact = scalar == expected
    try:
        exact = exact or Decimal(scalar) == Decimal(expected)
    except InvalidOperation:
        pass
    exact = exact and final["status"] == "answered"
    asked = premature = None
    if w["kind"] == "missing":
        first = xs[0]
        words = (first["answer"] or "").casefold()
        term = {
            "row": r"\b(row|record|entry|index|line)s?\b",
            "column": r"\b(column|field|header|attribute)s?\b",
            "date": r"\b(date|day|when|month|year)\b",
            "zone": r"\b(zone|timezone|region|location|city|where)\b",
            "document": r"\b(file|csv|document|spreadsheet|table|dataset)s?\b",
        }[w["missing"]]
        request_cue = (
            r"\?|\b(specify|provide|tell me|clarify|let me know|which|need to know)\b"
        )
        premature = (
            any("call" in t for t in first["turns"])
            or first["status"] == "invalid_call"
        )
        asked = (
            first["status"] == "answered"
            and not premature
            and bool(re.search(term, words) and re.search(request_cue, words))
        )
    return {
        "success": bool(path and exact and asked is not False),
        "path": path,
        "answer": answer,
        "exact": exact,
        "completed": all(x["status"] == "answered" for x in xs),
        "asked": asked,
        "premature": premature,
    }


def outcome(spec, material, data, validation):
    panel = material["rows"]
    groups = {}
    domains = spec.get("domains", ["ecology", "project"])
    for k in ("all", "complete", "missing", "public", *domains):
        ix = [
            i for i, w in enumerate(panel) if k in ("all", w["kind"], w.get("domain"))
        ]
        arms = {}
        for side, rows in data.items():
            subset = [rows[i] for i in ix]
            arms[side] = {
                "n": len(ix),
                **{
                    key: sum(bool(r["score"][key]) for r in subset)
                    for key in (
                        "success",
                        "path",
                        "answer",
                        "exact",
                        "completed",
                        "asked",
                        "premature",
                    )
                },
                "seconds": sum(r["seconds"] for r in subset),
                "first": sum(r["first"] for r in subset),
                "peak": max(r["peak"] for r in subset),
            }
        diffs = [
            int(data["candidate"][i]["score"]["success"])
            - int(data["control"][i]["score"]["success"])
            for i in ix
        ]
        groups[k] = {
            **arms,
            "gains": diffs.count(1),
            "losses": diffs.count(-1),
            "p": exact_mcnemar(diffs.count(1), diffs.count(-1)),
        }
    pairs = {}
    for name, rows in data.items():
        pairs[name] = [
            all(
                rows[i]["score"]["success"]
                for i, w in enumerate(panel)
                if w.get("anchor") == a
            )
            for a in range(16)
        ]
    diff = [
        int(a) - int(b)
        for a, b in zip(pairs["candidate"], pairs["control"], strict=True)
    ]
    pair_p = exact_mcnemar(diff.count(1), diff.count(-1))
    ref = unpack(Path(spec["initial"]) / "candidate.json.gz")
    keep = {r["id"] for r in ref if r["score"]["success"]}
    losses = [
        r["id"]
        for r in data["candidate"]
        if r["id"] in keep and not r["score"]["success"]
    ]
    state_fit = "state_width" in spec
    if state_fit:
        improved = sum(
            all(
                (v["probabilities"][k] >= 0.5) == known
                for k, known in v["known"].items()
            )
            for v in validation
        )
    else:
        improved = sum(
            (v["fitted"][0] - v["fitted"][1]) > (v["parent"][0] - v["parent"][1])
            for v in validation
        )
    a, b = groups["all"]["candidate"], groups["all"]["control"]
    passed = (
        sum(diff) >= 6
        and pair_p <= 0.05
        and sum(pairs["candidate"]) >= 8
        and all(groups[d]["candidate"]["success"] >= 4 for d in domains)
        and groups["missing"]["candidate"]["asked"] >= 10
        and groups["missing"]["candidate"]["premature"] <= 2
        and not losses
        and all(
            groups[k]["candidate"]["answer"] >= groups[k]["control"]["answer"]
            for k in ("complete", "missing", "public")
        )
        and a["seconds"] < b["seconds"]
        and a["peak"] < b["peak"]
        and improved >= spec.get("validation_threshold", 32)
        and (
            not state_fit
            or (
                a["first"] <= b["first"]
                and all(
                    groups[d]["candidate"]["completed"]
                    >= groups[d]["control"]["completed"]
                    for d in domains
                )
            )
        )
    )
    return {
        "passed": passed,
        "groups": groups,
        "pairs": pairs,
        "pair_gains": diff.count(1),
        "pair_losses": diff.count(-1),
        "pair_p": pair_p,
        "public_losses": losses,
        (
            "validation_all_fields_exact" if state_fit else "validation_improved"
        ): improved,
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    material = read(spec["material"])
    # Reconstruct the registered scalar independently of the live tool callbacks.
    sheets = {
        name: list(csv.DictReader(io.StringIO(record["text"], newline="")))
        for name, record in material["sheets"].items()
    }
    anchors = {}
    for item in material["rows"]:
        if item["kind"] == "public":
            continue
        args = item["call"]["arguments"]
        if item.get("domain") == "zones":
            from datetime import datetime
            from zoneinfo import ZoneInfo

            noon = datetime.fromisoformat(args["date"] + "T12:00:00").replace(
                tzinfo=ZoneInfo(args["zone"])
            )
            offset = (
                noon.dst()
                if item["call"]["name"] == "daylight_minutes"
                else noon.utcoffset()
            )
            scalar = int(offset.total_seconds() / 60)
        else:
            records = sheets[args["document"]]
            scalar = (
                len(records)
                if item["call"]["name"] == "csv_row_count"
                else records[args["row"] - 1][args["column"]]
            )
        assert scalar == item["gold"]
        anchors.setdefault(item["anchor"], []).append(item)
    assert len(anchors) == 16 and all(
        {w["kind"] for w in pair} == {"complete", "missing"}
        and len(pair) == 2
        and pair[0]["call"] == pair[1]["call"]
        for pair in anchors.values()
    )
    meta = read(root / "fit.json")
    assert meta["weights"] == digest(root / "tool-adapters.safetensors") and meta[
        "training"
    ] == digest(Path(spec["training"]))
    assert meta["updates"] == spec["updates"] and meta["passes"] == 2
    validation = read(root / "validation.json")
    if "state_width" in spec:
        from .dialogue_state_fit import fields

        frozen = read(spec["training"])
        assert meta["head"] == digest(root / "state-head.safetensors")
        assert meta["trainable"] == 802816 and meta["turns"] == len(frozen["examples"])
        assert len(validation) == 48 and all(
            r["id"] == w["id"]
            and r["known"] == w["known"]
            and r["known"].keys() == r["probabilities"].keys()
            and all(0 <= p <= 1 for p in r["probabilities"].values())
            for r, w in zip(validation, frozen["validation"], strict=True)
        )
        tokenizer = load_tokenizer(local_snapshot())
        for w in frozen["examples"] + frozen["validation"]:
            pos, labels = fields(tokenizer, w)
            assert len(pos) == len(labels)
    else:
        prefs = read(spec["training"])["preferences"]
        reference = read(root / "reference.json")
        assert reference["parent"] == spec["initial_sha256"] and len(
            reference["scores"]
        ) == len(prefs)
        assert [v["index"] for v in validation] == [
            i for i, p in enumerate(prefs) if p["split"] == "validation"
        ]
        assert all(
            v["parent"] == reference["scores"][v["index"]]
            and v["source_line"] == prefs[v["index"]]["source_line"]
            for v in validation
        )
    tok = load_tokenizer(local_snapshot())
    data = {}
    with documents(material) as files:
        for s in ("candidate", "control"):
            rows = unpack(root / (s + ".json.gz"))
            for w, r in zip(material["rows"], rows, strict=True):
                tools = tools_for(w, files)
                msgs = initial_messages("", tools)[:1]
                assert (
                    r["id"] == w["id"]
                    and r["exchanges"][0]["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                for q, tr in zip(w["users"], r["exchanges"], strict=True):
                    msgs.append({"role": "user", "content": q})
                    replay_exchange(tok, tools, msgs, tr, spec)
                    if "state_width" in spec:
                        expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
                        if s == "candidate":
                            expected.add((23, 1))
                        for t in tr["turns"]:
                            assert {
                                (x["layer"], x["turn"]) for x in t["kv_history"]
                            } == expected
                            assert all(
                                x["retained"]
                                == x["offset"]
                                == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                                for x in t["kv_history"]
                            )
                assert (
                    r["score"] == grade(w, r["exchanges"])
                    and 0 < r["first"] <= r["seconds"]
                    and r["peak"] > 0
                )
            data[s] = rows
    assert decision == outcome(spec, material, data, validation)


def run(name):
    from ..evaluation.language7 import base

    spec = registered(name)
    parent = Path(spec["initial"])
    assert (
        digest(parent / "checksums.json") == spec["initial_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    material = read(spec["material"])
    b = Budget(name, spec["limits"])
    data = {}
    try:
        with documents(material) as files:
            for s in ("candidate", "control"):
                if s == "candidate":
                    model, tok = restore(parent)
                    if (root / "fit.json").exists():
                        assert read(root / "fit.json")["weights"] == digest(
                            root / "tool-adapters.safetensors"
                        )
                        check_weights(
                            model, mx.load(str(root / "tool-adapters.safetensors"))
                        )
                    else:
                        if "state_width" in spec:
                            from .dialogue_state_fit import train as fit_state

                            fit_state(model, tok, spec, root, b)
                        else:
                            train(model, tok, spec, root, b)
                else:
                    model, tok = base()[:2]
                model.eval()
                path = root / (s + ".json.gz")
                arm = unpack(path) if path.exists() else []
                panel = material["rows"]
                assert [r["id"] for r in arm] == [w["id"] for w in panel[: len(arm)]]
                for w in panel[len(arm) :]:
                    b.check()
                    mx.synchronize()
                    mx.reset_peak_memory()
                    begin = perf_counter()
                    tools = tools_for(w, files)
                    history = initial_messages("", tools)[:1]
                    xs = [
                        session(
                            model,
                            tok,
                            q,
                            tools,
                            max_tokens=spec["max_tokens"],
                            max_turns=spec["max_turns"],
                            _messages=history,
                            progress=lambda _: b.check(),
                            record_kv_history="state_width" in spec,
                        )
                        for q in w["users"]
                    ]
                    mx.synchronize()
                    elapsed = perf_counter() - begin
                    assert xs[0]["turns"][0]["prompt_ids"] == w["initial_ids"]
                    arm.append(
                        {
                            "id": w["id"],
                            "exchanges": xs,
                            "score": grade(w, xs),
                            "seconds": elapsed,
                            "first": sum(x["turns"][0]["ttft_s"] for x in xs),
                            "peak": mx.get_peak_memory(),
                        }
                    )
                    if len(arm) % 6 == 0 or len(arm) == len(panel):
                        packed(path, arm)
                        print(
                            s,
                            len(arm),
                            sum(v["score"]["success"] for v in arm),
                            flush=True,
                        )
                data[s] = arm
                del model, tok
                gc.collect()
                mx.clear_cache()
        save(
            root / "decision.json",
            outcome(spec, material, data, read(root / "validation.json")),
        )
    finally:
        b.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
