"""Fit shared depth on official Qwen messages and run two native decoders through real callbacks."""

import argparse
import gc
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
from jsonschema import ValidationError, validate
from mlx_lm.utils import load_tokenizer

from .. import package_tools, qwen_tools
from ..evaluation import executable_tools, program_questions
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import read, save, digest
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_model import Assistant
from ..tool_session import compact
from .core import Budget
from .native_function_data import catalog, token_pair
from .registered import registered
from .tool_fit import build, fit


def optimize(spec, root, budget):
    model, tok = build(spec, 2)
    with TemporaryDirectory(dir=root.parent) as directory:

        def prepare(net, tokenizer, material, budget):
            bank = []
            for row in material["examples"]:
                budget.check()
                prefix, target = token_pair(tokenizer, row)
                hidden = net.prefix_hidden(mx.array([prefix + target[:-1]]))[0]
                path = Path(directory) / f"{len(bank)}.safetensors"
                mx.save_safetensors(str(path), {"hidden": hidden})
                bank.append((path, len(prefix) - 1, target))
            return bank

        fit(model, tok, spec, root, budget, prepare=prepare)


def executions(tr):
    calls = []
    for t in tr["turns"]:
        if "observation" in t:
            calls.append(dict(call=t["call"], observation=t["observation"]))
        elif "observations" in t:
            calls.extend(
                dict(call=c, observation=o)
                for c, o in zip(t["calls"], t["observations"], strict=True)
            )
    return calls


def grade(row, i, tr):
    events = executions(tr)
    if row["family"] == "public":
        return executable_tools.score(row, dict(tr, turns=events))
    if row["family"] in ("ASDiv", "SVAMP"):
        return program_questions.score(row, dict(tr, turns=events))
    expected = row["answers"][i]
    actual = (tr["answer"] or "").strip()
    complete = tr["status"] == "answered"
    path = [e["call"] for e in events] == [row["calls"][i]] and str(
        next(iter(events[0]["observation"].values()))
    ) == expected
    # Credit prose answers separately; strict success requires an unambiguous singleton.
    import re

    numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)*(?![\w]|\.\d)", actual)
    content = numbers == [expected] or actual == expected
    return dict(
        success=path and content and complete,
        path=path,
        correct_answer=content,
        completed=complete,
    )


def evaluate(spec, root, panel, side, budget):
    actor = None
    rows = []
    for w in panel["rows"]:
        tools = catalog(w)
        chat = None
        requests = []
        for i, text in enumerate(w["questions"]):
            mx.synchronize()
            mx.reset_peak_memory()
            start = perf_counter()
            if actor is None:
                actor = (
                    Assistant(spec["checkpoint"], tool_adapter=root, tool_format="qwen")
                    if side == "candidate"
                    else DenseAssistant(True, "qwen")
                )
            if chat is None:
                chat = actor.conversation(tools)
            stamps = {}

            def observe(consumed, timestamps=stamps, at=start):
                if consumed not in timestamps:
                    timestamps[consumed] = perf_counter() - at
                budget.check()

            tr = chat.respond(
                text,
                max_tokens=spec["max_tokens"],
                max_turns=spec["max_turns"],
                reuse_prefix=True,
                record_kv_history=True,
                progress=observe,
            )
            mx.synchronize()
            requests.append(
                dict(
                    trajectory=tr,
                    wall_s=perf_counter() - start,
                    first_s=stamps[len(tr["turns"][0]["prompt_ids"])],
                    peak=mx.get_peak_memory(),
                    score=grade(w, i, tr),
                )
            )
        assert requests[0]["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
        rows.append(dict(id=w["id"], family=w["family"], requests=requests))
        if len(rows) % 10 == 0:
            print(side, len(rows), flush=True)
    return rows


def summarize(rows):
    requests = [q for r in rows for q in r["requests"]]
    return dict(
        dialogues=len(rows),
        strict=sum(all(q["score"]["success"] for q in r["requests"]) for r in rows),
        requests=len(requests),
        answers=sum(q["score"]["correct_answer"] for q in requests),
        grounded=sum(q["score"]["success"] for q in requests),
        completed=sum(q["score"]["completed"] for q in requests),
        wall_s=sum(q["wall_s"] for q in requests),
        first_s=sum(q["first_s"] for q in requests),
        peak=max(q["peak"] for q in requests),
        output_tokens=sum(
            len(t["token_ids"]) for q in requests for t in q["trajectory"]["turns"]
        ),
    )


def outcome(spec, data):
    by = {
        s: {
            f: summarize([r for r in v if f == "all" or r["family"] == f])
            for f in ["all", "files", "packages", "public", "ASDiv", "SVAMP"]
        }
        for s, v in data.items()
    }
    good = {
        s: {r["id"] for r in arm if all(q["score"]["success"] for q in r["requests"])}
        for s, arm in data.items()
    }
    a, b = good["candidate"], good["control"]
    wins = len(a - b)
    losses = len(b - a)
    p = exact_mcnemar(wins, losses)
    prior = {
        r["id"]
        for r in unpack(Path(spec["public"]) / "candidate.json.gz")
        if r["score"]["success"]
    }
    x, y = by["candidate"], by["control"]
    service = sum(x[f]["strict"] for f in ("packages", "files"))
    gates = dict(
        strict=wins - losses >= 8 and p <= 0.05,
        field=service >= 6
        and all(x[f]["strict"] >= y[f]["strict"] for f in ("packages", "files")),
        public=prior <= a,
        fresh=all(x[f]["answers"] >= y[f]["answers"] for f in ("ASDiv", "SVAMP")),
        wall=x["all"]["wall_s"] <= y["all"]["wall_s"],
        first=x["all"]["first_s"] <= y["all"]["first_s"],
        peak=x["all"]["peak"] <= y["all"]["peak"],
    )
    return dict(
        passed=all(gates.values()),
        gates=gates,
        groups=by,
        wins=wins,
        losses=losses,
        p_exact=p,
        previous_public_lost=sorted(prior - a),
    )


def replay(tok, w, tools, history, request, spec, side):
    tr = request["trajectory"]
    decl = qwen_tools.declarations(tools)
    funcs = {t.name: t for t in tools}
    shapes = {t.name: t.definition() for t in tools}
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    answer = None
    status = "turn_limit"
    assert (
        0 < request["first_s"] <= request["wall_s"]
        and 0 < tr["seconds"] <= request["wall_s"]
        and request["peak"] > 0
    )
    assert 1 <= len(tr["turns"]) <= spec["max_turns"]
    for j, t in enumerate(tr["turns"]):
        assert t["prompt_ids"] == tok.encode(
            qwen_tools.render(tok, history, decl), add_special_tokens=False
        )
        ids = t["token_ids"]
        length = len(t["prompt_ids"]) + len(ids) - 1
        assert 0 < len(ids) <= spec["max_tokens"] and not any(
            i in stops for i in ids[:-1]
        )
        assert 0 < t["ttft_s"] <= t["seconds"]
        histories = t["kv_history"]
        sites = {(v["layer"], v["turn"]) for v in histories}
        assert sites == {(i, 0) for i in (3, 7, 11, 15, 19, 23)} | (
            {(23, 1)} if side == "candidate" else set()
        )
        assert all(v["retained"] == v["offset"] == length for v in histories)
        if ids[-1] not in stops:
            assert len(ids) == spec["max_tokens"]
            status = "token_limit"
        else:
            thought, sep, visible = tok.decode(ids[:-1]).partition("</think>")
            visible = visible.strip()
            if not sep:
                status = "unfinished_thinking"
            else:
                assert t["visible"] == visible
                try:
                    calls = qwen_tools.parse(visible, shapes)
                    for c in calls:
                        validate(c["arguments"], funcs[c["name"]].parameters)
                except (ValueError, ValidationError):
                    status = "invalid_call"
                    history.append(dict(role="assistant", content=visible))
                else:
                    if calls:
                        observed = []
                        for c in calls:
                            try:
                                r = funcs[c["name"]].callback(**c["arguments"])
                            except Exception as e:
                                r = {"error": type(e).__name__}
                            observed.append(r)
                        assert executions({"turns": [t]}) == [
                            dict(call=c, observation=o)
                            for c, o in zip(calls, observed, strict=True)
                        ]
                        history.append(
                            qwen_tools.assistant_message(
                                calls,
                                visible.partition("<tool_call>")[0].strip(),
                                thought.strip(),
                            )
                        )
                        history.extend(
                            dict(role="tool", content=compact(o)) for o in observed
                        )
                        continue
                    answer = visible
                    status = "answered" if answer else "empty_answer"
                    if answer:
                        history.append(dict(role="assistant", content=answer))
        assert j == len(tr["turns"]) - 1 and not executions({"turns": [t]})
    assert (status, answer) == (tr["status"], tr["answer"])


def verify(root, decision):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    panel = read(spec["material"])
    training = read(spec["training"])
    meta = read(root / "fit.json")
    tok = load_tokenizer(local_snapshot())
    assert all(digest(Path(k)) == v for k, v in training["sources"].items())
    assert meta["training"] == digest(Path(spec["training"])) and meta[
        "weights"
    ] == digest(root / "tool-adapters.safetensors")
    assert (
        meta["trainable"] == 802816
        and meta["passes"] == 2
        and meta["updates"] == spec["updates"]
    )
    lengths = [len(token_pair(tok, r)[1]) for r in training["examples"]]
    rng = random.Random(spec["fit_seed"])
    assert (
        len(lengths) == meta["turns"]
        and sum(sum(rng.sample(lengths, spec["batch"])) for _ in range(spec["updates"]))
        == meta["targets"]
    )
    data = {}
    for side in ("candidate", "control"):
        arm = unpack(root / (side + ".json.gz"))
        data[side] = arm
        for w, r in zip(panel["rows"], arm, strict=True):
            assert (w["id"], w["family"]) == (r["id"], r["family"])
            history = []
            tools = catalog(w, panel)
            assert (
                r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                == w["initial_ids"]
            )
            for i, (text, request) in enumerate(
                zip(w["questions"], r["requests"], strict=True)
            ):
                history.append(dict(role="user", content=text))
                replay(tok, w, tools, history, request, spec, side)
                assert grade(w, i, request["trajectory"]) == request["score"]
    assert outcome(spec, data) == decision


def run(name):
    spec = registered(name)
    parent = Path(spec["release"])
    assert (
        read(parent / "decision.json")["passed"]
        and digest(parent / "checksums.json") == spec["prerequisite_sha256"]
    )
    assert digest(Path(spec["public"]) / "checksums.json") == spec["public_sha256"]
    assert read(parent / "export-files.json") == read(
        Path(spec["checkpoint"]) / "tool-export-files.json"
    )
    assert all(
        digest(Path(spec["checkpoint"]) / k) == v
        for k, v in read(parent / "export-files.json").items()
    )
    panel = read(spec["material"])
    assert package_tools.snapshot() == panel["environment"] and all(
        digest(Path(k)) == v["sha256"] for k, v in panel["files"].items()
    )
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            optimize(spec, root, budget)
        data = {}
        for side in ("candidate", "control"):
            gc.collect()
            mx.clear_cache()
            dest = root / (side + ".json.gz")
            if not dest.exists():
                packed(dest, evaluate(spec, root, panel, side, budget))
            data[side] = unpack(dest)
        save(root / "decision.json", outcome(spec, data))
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
