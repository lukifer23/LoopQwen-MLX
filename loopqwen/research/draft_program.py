"""Learn to translate the ordinary model's own arithmetic reasoning into executable calls."""

import argparse
import gc
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from jsonschema import ValidationError, validate
from mlx_lm.utils import load_tokenizer

from ..calculator_tool import calculate, tool
from ..deliberation import ordinary_view
from ..draft_program import DraftAction, draft
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..evaluation.program_questions import INSTRUCTION, score
from ..evaluation.statistics import exact_mcnemar
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import load_base, local_snapshot
from ..tool_model import quantize_prefix, tool_view
from ..tool_session import (
    compact,
    decode_action,
    initial_messages,
    prompt,
)
from ..tool_session import (
    run as session,
)
from .core import Budget, atom
from .raw_distillation import check_weights
from .registered import registered
from .tool_fit import fit


def boundaries(tok):
    stops = set(tok.eos_token_ids) | {atom(tok, "<|im_end|>")}
    return stops, stops | {atom(tok, "</think>"), atom(tok, "<|im_start|>")}


def encode(tok, messages):
    return tok.encode(prompt(tok, messages), add_special_tokens=False)


def build(spec):
    text, tok, _ = load_base()
    quantize_prefix(text, spec["start_layer"], spec["prefix_bits"])
    mx.random.seed(spec["fit_seed"])
    model = tool_view(text, spec["start_layer"], spec["rank"], spec["passes"])
    model.eval()
    return model, tok


def validate_draft(r, tokens, limit, ends):
    assert 0 <= len(tokens) <= limit and ends.isdisjoint(tokens)
    if len(tokens) < limit:
        assert r["proposed_boundary"] in ends
    else:
        assert r["proposed_boundary"] is None


def prepare(model, ordinary, tok, rows, spec, out, budget):
    ends = boundaries(tok)[1]
    dst = out / "drafts.json.gz"
    records = unpack(dst) if dst.is_file() else []
    eos = atom(tok, "<|im_end|>")
    bank = []

    def add(prefix, target):
        reply = tok.encode(target, add_special_tokens=False) + [eos]
        h = model.prefix_hidden(mx.array([prefix + reply[:-1]]))[0]
        mx.eval(h)
        bank.append((h, len(prefix) - 1, reply))

    for i, w in enumerate(rows["rows"]):
        budget.check()
        msgs = initial_messages(w["question"] + INSTRUCTION, [tool()])
        ids = encode(tok, msgs)
        if i == len(records):
            tokens, boundary, _ = draft(
                ordinary, ids, spec["draft_tokens"], ends, lambda _: budget.check()
            )
            records.append(
                {
                    "id": w["id"],
                    "sha256": fingerprint(ids),
                    "ids": tokens,
                    "proposed_boundary": boundary,
                }
            )
            if (i + 1) % 32 == 0 or i + 1 == len(rows["rows"]):
                packed(dst, records)
                print("self-drafted training questions", i + 1, flush=True)
        r = records[i]
        assert r["id"] == w["id"] and r["sha256"] == fingerprint(ids)
        validate_draft(r, r["ids"], spec["draft_tokens"], ends)
        steps = w["steps"]
        obs = calculate(steps)
        assert Fraction(obs["value"]) == Fraction(w["gold"])
        call = compact({"name": "calculate", "arguments": {"steps": steps}})
        add(
            ids + r["ids"],
            "\nThe needed operations are "
            + ", then ".join(steps)
            + ".\n</think>\n\n"
            + call,
        )
        msgs.extend(
            (
                {"role": "assistant", "content": call},
                {"role": "tool", "content": compact(obs)},
            )
        )
        add(encode(tok, msgs), "</think>\n\n" + w["gold"])
    assert len(records) == len(rows["rows"])
    return bank


def outcome(data, panel, spec):
    result = {}
    for src in ["all", *sorted({w["source"] for w in panel})]:
        ix = [i for i, w in enumerate(panel) if src in ("all", w["source"])]
        group = {}
        hits = {}
        for side, arr in data.items():
            rows = [arr[i] for i in ix]
            turns = [t for r in rows for t in r["trajectory"]["turns"]]
            hits[side] = [int(r["score"]["correct_answer"]) for r in rows]
            group[side] = {
                **{k: sum(r["score"][k] for r in rows) for k in rows[0]["score"]},
                "n": len(rows),
                "seconds": sum(r["trajectory"]["seconds"] for r in rows),
                "first": sum(r["trajectory"]["turns"][0]["ttft_s"] for r in rows),
                "per_turn_first": sum(t["ttft_s"] for t in turns),
                "tokens": sum(len(t["token_ids"]) for t in turns),
                "draft_tokens": sum(
                    len(r["draft"]["ids"]) for r in rows if r["draft"] is not None
                ),
                "peak": max(t["peak_bytes"] for t in turns),
            }
        delta = [a - b for a, b in zip(hits["candidate"], hits["control"], strict=True)]
        g, l = delta.count(1), delta.count(-1)
        result[src] = {**group, "gains": g, "losses": l, "p": exact_mcnemar(g, l)}
    all_ = result["all"]
    a, b = all_["candidate"], all_["control"]
    passed = (
        all_["gains"] - all_["losses"] >= spec["min_gain"]
        and all_["p"] <= 0.05
        and a["success"] >= b["success"] + spec["grounded_gain"]
        and all(a[k] <= b[k] for k in ("seconds", "peak", "first"))
        and all(
            r["gains"] - r["losses"] >= 2
            and all(
                r["candidate"][k] >= r["control"][k]
                for k in ("completed", "strict_answer", "success", "path")
            )
            for name, r in result.items()
            if name != "all"
        )
    )
    return {"passed": passed, "groups": result}


def replay(tok, w, traj, spec):
    calc = tool()
    msgs = initial_messages(w["user_request"], [calc])
    stops, _ = boundaries(tok)
    answer = None
    status = "turn_limit"
    turns = traj["turns"]
    assert (
        1 <= len(turns) <= spec["max_turns"]
        and turns[0]["prompt_ids"] == w["prompt_ids"]
    )
    for i, t in enumerate(turns):
        assert t["prompt_ids"] == encode(tok, msgs)
        ids = t["token_ids"]
        assert 0 < len(ids) <= spec["cap"] and stops.isdisjoint(ids[:-1])
        assert (
            traj["seconds"] >= t["seconds"] >= t["ttft_s"] > 0 and t["peak_bytes"] > 0
        )
        raw = tok.decode(ids[:-1])
        if ids[-1] not in stops:
            assert len(ids) == spec["cap"]
            status = "token_limit"
        elif "</think>" not in raw:
            status = "unfinished_thinking"
        else:
            visible = raw.split("</think>", 1)[1].strip()
            assert visible == t["visible"]
            try:
                call = decode_action(visible)
                if call is None:
                    answer = visible
                    status = "answered" if visible else "empty_answer"
                else:
                    assert t["call"] == call
                    if call["name"] != calc.name:
                        raise ValueError("Unknown tool")
                    validate(call["arguments"], calc.parameters)
                    try:
                        obs = calc.callback(**call["arguments"])
                    except (
                        ValueError,
                        ArithmeticError,
                        SyntaxError,
                        RecursionError,
                    ) as exc:
                        obs = {"error": type(exc).__name__}
                    assert obs == t["observation"]
                    msgs.extend(
                        (
                            {"role": "assistant", "content": visible},
                            {"role": "tool", "content": compact(obs)},
                        )
                    )
                    continue
            except (ValueError, ValidationError):
                status = "invalid_call"
        assert i + 1 == len(turns) and "observation" not in t
    assert traj["answer"] == answer and traj["status"] == status


def verify(root, d):
    spec = read(root / "registration.json")
    fit_meta = read(root / "fit.json")
    training = read(spec["training"])["rows"]
    panel = read(spec["material"])["rows"]
    tok = load_tokenizer(local_snapshot())
    ends = boundaries(tok)[1]
    assert fit_meta["weights"] == digest(
        root / "tool-adapters.safetensors"
    ) and fit_meta["training"] == digest(Path(spec["training"]))
    assert (
        fit_meta["updates"] == spec["updates"]
        and fit_meta["turns"] == 2 * len(training)
        and fit_meta["passes"] == spec["passes"]
    )
    for w, r in zip(training, unpack(root / "drafts.json.gz"), strict=True):
        ids = encode(tok, initial_messages(w["question"] + INSTRUCTION, [tool()]))
        assert (
            r["id"] == w["id"]
            and r["sha256"] == fingerprint(ids)
            and Fraction(calculate(w["steps"])["value"]) == Fraction(w["gold"])
        )
        validate_draft(r, r["ids"], spec["draft_tokens"], ends)
    data = {}
    for s in ("candidate", "control"):
        arr = unpack(root / (s + ".json.gz"))
        for w, r in zip(panel, arr, strict=True):
            tr = r["trajectory"]
            assert r["id"] == w["id"] and fingerprint(w["prompt_ids"]) == w["sha256"]
            replay(tok, w, tr, spec)
            assert score(w, tr) == r["score"]
            if s == "candidate":
                receipt = r["draft"]
                n = len(receipt["ids"])
                validate_draft(receipt, receipt["ids"], spec["draft_tokens"], ends)
                assert tr["turns"][0]["token_ids"][:n] == receipt["ids"]
            else:
                assert r["draft"] is None
        data[s] = arr
    assert outcome(data, panel, spec) == d


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    panel = read(spec["material"])["rows"]
    b = Budget(name, spec["limits"])
    data = {}
    try:
        for s in ("candidate", "control"):
            decoder = None
            if s == "candidate":
                model, tok = build(spec)
                ordinary = ordinary_view(model)
                if (out / "fit.json").exists():
                    meta = read(out / "fit.json")
                    path = out / "tool-adapters.safetensors"
                    assert meta["weights"] == digest(path) and meta[
                        "training"
                    ] == digest(Path(spec["training"]))
                    check_weights(model, mx.load(str(path)))
                else:

                    def self_drafts(net, tokenizer, rows, budget, ordinary=ordinary):
                        return prepare(
                            net, ordinary, tokenizer, rows, spec, out, budget
                        )

                    fit(model, tok, spec, out, b, prepare=self_drafts)
                    del self_drafts
                model.eval()
                decoder = DraftAction(ordinary, spec["draft_tokens"])
            else:
                model, tok = base()[:2]
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            dest = out / (s + ".json.gz")
            arm = unpack(dest) if dest.exists() else []
            assert [r["id"] for r in arm] == [w["id"] for w in panel[: len(arm)]]
            for w in panel[len(arm) :]:
                b.check()
                tr = session(
                    model,
                    tok,
                    w["user_request"],
                    [tool()],
                    max_tokens=spec["cap"],
                    max_turns=spec["max_turns"],
                    first_turn_decoder=decoder,
                    progress=lambda _: b.check(),
                )
                assert tr["turns"][0]["prompt_ids"] == w["prompt_ids"]
                arm.append(
                    {
                        "id": w["id"],
                        "trajectory": tr,
                        "score": score(w, tr),
                        "draft": decoder.receipt if decoder else None,
                    }
                )
                if len(arm) % 8 == 0 or len(arm) == len(panel):
                    packed(dest, arm)
                    print(
                        s,
                        len(arm),
                        sum(r["score"]["correct_answer"] for r in arm),
                        flush=True,
                    )
            data[s] = arm
            if decoder:
                del ordinary, decoder
            del model, tok
            gc.collect()
            mx.clear_cache()
        save(out / "decision.json", outcome(data, panel, spec))
    finally:
        b.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
