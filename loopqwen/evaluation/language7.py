"""Calibrate on untouched Base before comparing fixed-depth deployments."""

import gc
import json
import random
from pathlib import Path
from time import perf_counter
import mlx.core as mx
from .. import REVISION
from ..data import graphs, pointer5, reasoning
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import load_base
from ..looped_model import LoopedModel
from ..recurrence import Span
from ..training.exposure import ARM, OUT, PARENT, S
from .answers import numeric
from .mechanism import load, packed, unpack

ROOT = OUT / "language"
DEMOS = [
    ("What is 4 times 3 plus 1?", "4 × 3 = 12, and 12 + 1 = 13.", "13"),
    (
        "A path goes from 2 to 6 and from 6 to 1. Starting at 2, where are you after two links?",
        "The two successive rooms are 6 and 1.",
        "1",
    ),
]


def cohorts():
    rng = random.Random(S["calibration"]["seed"])
    exclude = {
        r["map_id"]
        for split in ("train", "development", "confirmation")
        for r in pointer5.rows(split)
    }
    for case in pointer5.rows("cases"):
        exclude.update(
            v["map_id"] for v in case.values() if isinstance(v, dict) and "map_id" in v
        )

    def item(q, g, family, **data):
        return dict(id=fingerprint(q), question=q, gold=str(g), family=family, **data)

    def pointer():
        while True:
            r = pointer5.instance(rng.sample(range(8), 8), rng.randrange(8))
            if r["map_id"] not in exclude and pointer5.cycle(r) > 3:
                break
        exclude.add(r["map_id"])
        edges = ", ".join(
            f"{i} leads to {r['map'][i]}" for i in rng.sample(range(8), 8)
        )
        q = f"Room links: {edges}. Start in room {r['start']} and follow exactly three links. What is the destination?"
        return item(q, r["states"][2], "pointer", map=r["map"], start=r["start"])

    def proc(i):
        g = (
            graphs.graph(rng, i % 2)
            if i % 3 == 2
            else reasoning.problem(
                rng, "arithmetic" if i % 3 == 0 else "modular", i % 2
            )
        )
        return item(
            g["question"],
            g["answer"],
            g["category"],
            generator_id=g["id"],
            certificate=g["trace"],
        )

    cal = [
        item("What is 18 plus 7?", 25, "arithmetic"),
        pointer(),
        proc(2),
        proc(0),
        pointer(),
        proc(4),
        proc(5),
        pointer(),
    ]
    pair = [x for i in range(12) for x in (pointer(), proc(i))]
    assert len({r["id"] for r in cal + pair}) == 32
    return cal, pair


def prompt_ids(t, row, protocol):
    def assistant_start(q):
        return t.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": q
                    + "\nReason briefly, then give a single number on the last line as Final answer: <number>.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    text = "".join(
        assistant_start(q)
        + thought
        + "\n</think>\n\nFinal answer: "
        + ans
        + "<|im_end|>\n"
        for q, thought, ans in DEMOS[: protocol["demos"]]
    ) + assistant_start(row["question"])
    ids = t.encode(text, add_special_tokens=False)
    if not text.endswith("<think>\n") or t.decode(ids) != text:
        raise ValueError("Chat control tokens failed exact round trip")
    return ids


def interpret(ids, opening, closing, eos, t):
    opens = [i for i, x in enumerate(ids) if x == opening]
    closes = [i for i, x in enumerate(ids) if x == closing]
    end = len(ids) - int(bool(ids) and ids[-1] in eos)
    visible_ids = ids[closes[0] + 1 : end] if len(closes) == 1 and not opens else []
    visible = t.decode(visible_ids)
    answer = numeric(visible) if visible_ids else None
    return dict(
        open_positions=opens,
        close_positions=closes,
        thinking_tokens=closes[0] if closes else end,
        visible_tokens=len(visible_ids),
        visible=visible,
        answer=answer,
    )


def trial(m, t, weights, r, p, out):
    ids = prompt_ids(t, r, p)
    identity = fingerprint(
        [
            weights,
            ids,
            r,
            p,
            digest(Path(__file__)),
            REVISION,
            digest(Path("loopqwen/evaluation/answers.py")),
        ]
    )
    if out.exists():
        old = unpack(out)
        if old["identity"] != identity:
            raise ValueError("Incompatible generation record")
        return old
    op, cl = [t.encode(x, add_special_tokens=False) for x in ("<think>", "</think>")]
    if len(op) != 1 or len(cl) != 1:
        raise ValueError("Thinking markers must be atomic")
    eos = t.eos_token_ids
    m.eval()
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    cache = m.make_cache()
    x = mx.array([ids])
    tokens = []
    while x.shape[1] > 128:
        mx.eval(m.hidden(x[:, :128], cache))
        x = x[:, 128:]
    z = m.project(m.hidden(x, cache)[:, -1])
    mx.eval(z)
    prefill = perf_counter() - start
    for i in range(p["cap"]):
        tok = mx.argmax(z, axis=-1).item()
        tokens.append(tok)
        if i == 0:
            ttft = perf_counter() - start
        if tok in eos or i + 1 == p["cap"]:
            break
        z = m.project(m.hidden(mx.array([[tok]]), cache)[:, -1])
    seconds = perf_counter() - start
    parsed = interpret(tokens, op[0], cl[0], eos, t)
    stop = "eos" if tokens[-1] in eos else "cap"
    if stop == "cap":
        parsed["answer"] = None
    status = (
        ("correct" if parsed["answer"] == numeric(r["gold"]) else "incorrect")
        if parsed["answer"] is not None
        else ("truncated" if stop == "cap" else "malformed")
    )
    result = dict(
        identity=identity,
        weights=weights,
        item=r,
        protocol=p,
        prompt_ids=ids,
        prompt_open_positions=[i for i, x in enumerate(ids) if x == op[0]],
        prompt_close_positions=[i for i, x in enumerate(ids) if x == cl[0]],
        opening_id=op[0],
        closing_id=cl[0],
        token_ids=tokens,
        text=t.decode(tokens),
        stop=stop,
        status=status,
        seconds=seconds,
        prefill_s=prefill,
        ttft_s=ttft,
        decode_s=seconds - ttft,
        peak_bytes=mx.get_peak_memory(),
        **parsed,
    )
    packed(out, result)
    return result


def stats(rows):
    n = len(rows)
    valid = sum(r["answer"] is not None for r in rows)
    correct = sum(r["status"] == "correct" for r in rows)
    return dict(
        n=n,
        valid=valid,
        correct=correct,
        accuracy=correct / n if n else None,
        conditional=correct / valid if valid else None,
        capped=sum(r["stop"] == "cap" for r in rows),
        tokens=sum(len(r["token_ids"]) for r in rows),
        seconds=sum(r["seconds"] for r in rows),
    )


def base():
    b, t, _ = load_base()
    return LoopedModel(b, Span(passes=1)), t, REVISION


def compare(cohort, p, arm, step, folder, *, floor=True):
    results = {k: [] for k in ("base", "candidate")}
    for block in (cohort[:6], cohort[6:]):
        if not block:
            continue
        half = len(block) // 2
        for name, items in (
            ("base", block[:half]),
            ("candidate", block),
            ("base", block[half:]),
        ):
            if name == "base":
                m, t, w = base()
            else:
                m, t, _, w = load(arm, step)
            for r in items:
                results[name].append(
                    trial(m, t, w, r, p, folder / name / (r["id"] + ".json.gz"))
                )
            del m, t
            gc.collect()
            mx.clear_cache()
        for a, b in zip(*results.values(), strict=True):
            if a["prompt_ids"] != b["prompt_ids"] or a["item"] != b["item"]:
                raise ValueError("Mismatched comparison")
        if floor and min(stats(xs)["valid"] for xs in results.values()) < 2:
            break
        if not floor and all(stats(xs)["correct"] == 0 for xs in results.values()):
            break
    return {
        k: {
            group: stats(
                [
                    r
                    for r in v
                    if group == "all"
                    or (r["item"]["family"] == "pointer") == (group == "pointer")
                ]
            )
            for group in ("all", "pointer", "other")
        }
        for k, v in results.items()
    }


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    cal, questions = cohorts()
    save(
        ROOT / "cohorts.json", dict(calibration=cal, comparison=questions, demos=DEMOS)
    )
    m, t, w = base()
    selected = None
    exam = {}
    for p in S["calibration"]["protocols"]:
        rows = [
            trial(m, t, w, r, p, ROOT / p["name"] / (r["id"] + ".json.gz")) for r in cal
        ]
        s = exam[p["name"]] = stats(rows)
        print(p["name"], s, flush=True)
        if s["valid"] >= 6 and 0 < s["correct"] < 8:
            selected = p
            break
    del m, t
    gc.collect()
    mx.clear_cache()
    result = dict(calibration=exam, selected=selected)
    save(ROOT / "decision.json", result)
    if selected:
        d = read(OUT / "decision.json")
        arm, step = (ARM, d["step"]) if d["installed"] else (PARENT, 480)
        pair = compare(questions, selected, arm, step, ROOT / "pair")
        result["pair"] = pair
        if (
            pair["candidate"]["pointer"]["correct"] - pair["base"]["pointer"]["correct"]
            >= 3
        ):
            from dataclasses import replace

            m, t, _, w = load(arm, step)
            m.span = replace(m.span, passes=1)
            result["bypass"] = stats(
                [
                    trial(
                        m,
                        t,
                        w + "/K1",
                        r,
                        selected,
                        ROOT / "bypass" / (r["id"] + ".json.gz"),
                    )
                    for r in questions
                ]
            )
            del m, t
            gc.collect()
            mx.clear_cache()
        from .gsm7 import reserved

        result["gsm"] = compare(
            reserved(ROOT), selected, arm, step, ROOT / "gsm", floor=False
        )
        save(ROOT / "decision.json", result)
