"""Obtain real training roll-ins and replay both supervision and callback arithmetic."""

import argparse
from pathlib import Path
import json
from time import monotonic

from loopqwen.calculator_tool import calculate, tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import MiddleAssistant
from loopqwen.qwen_tools import declarations
from loopqwen.research.native_function_data import token_pair
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_material import ROWS
from scripts.program_language_material import answer
from scripts.middle_depth_screen import SOURCE as DONOR_SCREEN

SOURCE = Path("research/sources/crossed-report-calculations/training.json.gz")
PLAN = Path("research/diagnostics/middle-depth/training-rollin.json")


def freeze():
    training = unpack(ROWS)
    clusters = {
        r["cluster"]
        for r in training
        if r["group"] in ("open", "deliver", "repair", "recovered")
    }
    reports = sorted(clusters, key=lambda s: fingerprint([4619, "rollin", s]))[:8]
    raw = unpack(SOURCE)
    chosen = []
    for report in reports:
        r = min(
            (r for r in raw if r["report"] == report and r["condition"] == "correct"),
            key=lambda r: r["question"],
        )
        assert calculate(r["correct_steps"]) == r["observed"] == r["true"]
        chosen.append(r["question"])
    save(
        PLAN,
        dict(
            source=digest(SOURCE),
            data=digest(ROWS),
            questions=chosen,
            clusters=reports,
            budget=256,
            turns=3,
            stage=32,
            method="Run the two actual early policies on these training reports. Retain every attempted trajectory and exact callback, including successful or failed provider execution. Replay the publisher-aligned operands. From unfinished deliberation take a genuine unclosed prefix of at most 32 emitted tokens and correct it with the verified program. After an actual callback train the appropriate next action or supported final response. Both arms receive the pooled corrections in the same order if an independently registered correction opens. Grade yield against all sixteen attempts.",
        ),
    )


def collect(arm, policy):
    cfg = read(PLAN)
    assert cfg["source"] == digest(SOURCE) and cfg["data"] == digest(ROWS)
    raw = {r["question"]: r for r in unpack(SOURCE) if r["condition"] == "correct"}
    with worker("fit", "training roll-in/" + arm) as guard:
        assistant = MiddleAssistant(BASE, policy)
        model, tok = assistant._tool_model, assistant.tokenizer
        for key in cfg["questions"]:
            r = raw[key]
            out = ROOT / "rollin" / arm / (fingerprint(key) + ".json.gz")
            if out.exists():
                old = unpack(out)
                assert old["policy"] == digest(policy / "adapters.safetensors") and old[
                    "plan"
                ] == digest(PLAN)
                continue
            start = monotonic()
            cmds = [tool()]
            assert r["tools"] == declarations(cmds)
            p = r["messages"][0]
            assert p["role"] == "user"
            gold = r["messages"][1]
            obs = calculate(r["correct_steps"])
            assert (
                obs == r["true"]
                and gold["tool_calls"][0]["function"]["arguments"]["steps"]
                == r["correct_steps"]
            )
            history = []
            tr = run(
                model,
                tok,
                p["content"],
                cmds,
                tool_format="qwen",
                max_tokens=cfg["budget"],
                max_turns=cfg["turns"],
                prefill_cache_state=True,
                record_kv_history=True,
                progress=guard,
                _messages=history,
            )
            corrections = []
            added_prefix = False
            for t in tr["turns"]:
                if "call" in t:
                    assert t["call"]["name"] == "calculate"
                    try:
                        replay = calculate(**t["call"]["arguments"])
                    except Exception as e:
                        replay = {"error": type(e).__name__}
                    assert replay == t["observation"]
                # A truncation before the visible answer is an independently
                # established failure, even if some underlying reasoning was sound.
                ids = t["token_ids"]
                if (
                    not added_prefix
                    and tr["status"] == "token_limit"
                    and t is tr["turns"][-1]
                    and 248069 not in ids[:32]
                ):
                    prefix = ids[:32]
                    reply = tok.encode(
                        " Let's check the report quantities against the requested calculation.\n</think>\n\n",
                        add_special_tokens=False,
                    )
                    gold_p, gold_y = token_pair(
                        tok, dict(messages=[p], tools=r["tools"], target=gold)
                    )
                    close = gold_y.index(248069)
                    reply = reply[: reply.index(248069)] + gold_y[close:]
                    corrections.append(
                        dict(
                            kind="unfinished",
                            p=t["prompt_ids"] + prefix,
                            y=reply,
                            paired_p=gold_p,
                            paired_y=gold_y,
                            actual_prefix=prefix,
                        )
                    )
                    added_prefix = True
            for i, msg in enumerate(history):
                if msg["role"] != "tool":
                    continue
                seen = json.loads(msg["content"])
                correct = seen == obs
                target = answer(r["reference"]) if correct else gold
                inputs, labels = token_pair(
                    tok,
                    dict(messages=history[: i + 1], tools=r["tools"], target=target),
                )
                shared = r["messages"] if correct else [p]
                base_p, base_y = token_pair(
                    tok, dict(messages=shared, tools=r["tools"], target=target)
                )
                assert labels == base_y
                corrections.append(
                    dict(
                        kind="delivery" if correct else "new calculation",
                        p=inputs,
                        y=labels,
                        paired_p=base_p,
                        paired_y=base_y,
                        observation=seen,
                    )
                )
            packed(
                out,
                dict(
                    question=key,
                    report=r["report"],
                    plan=digest(PLAN),
                    policy=digest(policy / "adapters.safetensors"),
                    seconds=monotonic() - start,
                    reference=r["reference"],
                    verified=obs,
                    steps=r["correct_steps"],
                    original_prompt=p["content"],
                    trajectory=tr,
                    history=history,
                    corrections=corrections,
                ),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "collect"))
    p.add_argument("--arm", choices=("loop", "ordinary"))
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    freeze() if a.action == "freeze" else collect(a.arm, a.policy)
