"""Unrestricted source-paired generation with physical callbacks and input hashes."""

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx

from loopqwen.calculator_tool import tool
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import MiddleAssistant, architecture, donor
from loopqwen.qwen_tools import declarations, render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker

SOURCE = Path("research/diagnostics/middle-depth/short-dev.json")
FROZEN = SOURCE.with_name("short-inputs.json")
WORLDS = ("original", "changed", "irrelevant")


def instance(r, world, folder):
    context = r.get("context", "")
    data = r.get("csv", "")
    if world != "original":
        k = "change" if world == "changed" else "irrelevant"
        if r.get("provider") == "csv":
            before, after = r["data_" + k]
            assert data.count(before) == 1
            data = data.replace(before, after)
        else:
            before, after = r[k]
            assert context.count(before) == 1
            context = context.replace(before, after)
    tools = []
    if r.get("provider") == "calculator":
        tools = [tool()]
    elif r.get("provider") == "csv":
        file = folder / r["document"]
        file.write_text(data)
        tools = csv_tools([file], keyed=True)
    question = (context + "\n\n" + r["question"]).strip() if context else r["question"]
    return question, tools


def prompt_ids(tok, msgs, tools):
    return tok.encode(render(tok, msgs, declarations(tools)), add_special_tokens=False)


def freeze():
    tok = tokenizer({"checkpoint": str(BASE)})
    rows = {}
    with TemporaryDirectory() as tmp:
        for r in read(SOURCE):
            for world in WORLDS:
                q, t = instance(r, world, Path(tmp))
                msgs = [dict(role="user", content=q)]
                rows[r["id"] + "/" + world] = fingerprint(prompt_ids(tok, msgs, t))
                if world == "original" and "followup" in r:
                    msgs.extend(
                        [
                            dict(role="assistant", content=r["shared"]),
                            dict(role="user", content=r["followup"]),
                        ]
                    )
                    rows[r["id"] + "/shared"] = fingerprint(prompt_ids(tok, msgs, t))
    save(
        FROZEN,
        dict(
            source=digest(SOURCE),
            ids=rows,
            protocol="Single source-blinded rubric review. Completed visible answer, all requested details and supported execution. An irrelevant edit keeps the original reference. Own histories include the real preceding trajectory, including a failed prefix; shared histories use the single registered first answer. Score the followups apart from the three source worlds. Keep all physical generations. Stage 0 and 32 use 128 tokens, terminal uses 128 and 512 per turn. Order rotates by source block.",
        ),
    )


def collect(args):
    cfg = read(FROZEN)
    assert cfg["source"] == digest(SOURCE)
    side = args.arm
    name = f"{side}-k{args.depth}" if args.depth else side
    folder = ROOT / "short" / args.stage / str(args.tokens) / name
    with (
        worker("short", str(folder) + "/" + str(args.block)) as guard,
        TemporaryDirectory() as tmp,
    ):
        if args.policy:
            assistant = MiddleAssistant(BASE, args.policy)
            model, tok = assistant._tool_model, assistant.tokenizer
            weights = read(args.policy / "registration.json")["weights"]
        else:
            text, tok = donor(BASE, "instruction" if side == "instruction" else "base")
            model = (
                architecture(text, 2, 4619)
                if side == "loop"
                else LoopedModel(text, Span(12, 4, 1))
            )
            model.eval()
            weights = "untouched-" + (
                "instruction" if side == "instruction" else "base"
            )
        if args.depth:
            model.span = replace(model.span, passes=args.depth)

        def take(r, world, question, tools, history):
            path = folder / (r["id"] + "-" + world + ".json.gz")
            if path.exists():
                existing = unpack(path)
                assert existing["weights"] == weights and existing[
                    "protocol"
                ] == digest(FROZEN)
                return existing["history_after"]
            before = list(history)
            sig = fingerprint(
                prompt_ids(tok, [*before, dict(role="user", content=question)], tools)
            )
            if world != "own":
                assert sig == cfg["ids"][r["id"] + "/" + world]
            times = []
            start = monotonic()

            def timer(i):
                guard()
                times.append([i, monotonic() - start])

            mx.reset_peak_memory()
            trace = run(
                model,
                tok,
                question,
                tools,
                max_tokens=args.tokens,
                max_turns=3,
                _messages=history,
                tool_format="qwen",
                record_kv_history=True,
                prefill_cache_state=True,
                progress=timer,
            )
            assert fingerprint(trace["turns"][0]["prompt_ids"]) == sig
            packed(
                path,
                dict(
                    source=r["id"],
                    family=r["family"],
                    world=world,
                    weights=weights,
                    protocol=digest(FROZEN),
                    initial=sig,
                    history_before=before,
                    history_after=history,
                    timing=times,
                    trace=trace,
                ),
            )
            return history

        subset = set(read(args.items)["items"]) if args.items else None
        for i, r in enumerate(read(SOURCE)):
            if i % 2 != args.block:
                continue
            for world in WORLDS:
                if subset is not None and r["id"] + "/" + world not in subset:
                    continue
                q, t = instance(r, world, Path(tmp))
                history = take(r, world, q, t, [])
                if world == "original" and "followup" in r:
                    take(r, "own", r["followup"], t, list(history))
                    take(
                        r,
                        "shared",
                        r["followup"],
                        t,
                        [
                            dict(role="user", content=q),
                            dict(role="assistant", content=r["shared"]),
                        ],
                    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "collect"))
    p.add_argument("--arm", choices=("loop", "ordinary", "instruction", "base"))
    p.add_argument(
        "--stage",
        choices=("initial", "early", "terminal", "simple", "corrected", "deployed", "depth"),
        default="initial",
    )
    p.add_argument("--block", choices=(0, 1), type=int, default=0)
    p.add_argument("--depth", type=int, choices=(1, 2, 3))
    p.add_argument("--policy", type=Path)
    p.add_argument("--tokens", type=int, choices=(128, 512, 4096), default=128)
    p.add_argument("--items", type=Path, help="Frozen source/world keys for the supplemental headroom comparison")
    a = p.parse_args()
    freeze() if a.action == "freeze" else collect(a)
