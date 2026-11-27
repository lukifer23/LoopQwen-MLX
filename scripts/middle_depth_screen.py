"""Freeze exact common openings and collect complete donor-screen trajectories."""
import argparse
from pathlib import Path

import mlx.core as mx
from loopqwen.calculator_tool import tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import donor
from loopqwen.looped_model import LoopedModel
from loopqwen.qwen_tools import declarations, render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker

SOURCE = Path("research/diagnostics/middle-depth/donor-screen.json")
INPUT = SOURCE.with_name("donor-ids.json")


def tools(row):
    return [tool()] if row.get("provider") == "calculator" else []


def freeze():
    tok = tokenizer({"checkpoint": str(BASE)})
    prompts = {}
    for row in read(SOURCE):
        msgs = [*row.get("history", []), {"role":"user","content":row["prompt"]}]
        prompts[row["source"]] = tok.encode(render(tok, msgs, declarations(tools(row))), add_special_tokens=False)
    save(INPUT, dict(source=digest(SOURCE), inputs=prompts, max_tokens=256, max_turns=3,
        decision="One point for a completed, supported answer satisfying every reference requirement. Two interleaved blocks alternate which donor runs first per source. Manual source review on blinded trajectories. Strictly more total credit selects instruction. Ties select Base."))


def collect(block, which):
    cfg = read(INPUT)
    assert cfg["source"] == digest(SOURCE)
    with worker("profile", f"screen/{block}/{which}") as guard:
        text, tok = donor(BASE, which)
        model = LoopedModel(text, Span(12,4,1))
        model.eval()
        for i, row in enumerate(read(SOURCE)):
            if i % 2 != block:
                continue
            path = ROOT / "screen" / f"{which}-{i}.json.gz"
            if path.exists():
                prev = unpack(path)
                assert prev["frozen"] == digest(INPUT)
                continue
            mx.reset_peak_memory()
            trace = run(model, tok, row["prompt"], tools(row), tool_format="qwen",
                _messages=list(row.get("history",[])), max_turns=3, max_tokens=256,
                record_kv_history=True, prefill_cache_state=True, progress=guard)
            ids = trace["turns"][0]["prompt_ids"]
            assert ids == cfg["inputs"][row["source"]]
            packed(path, dict(source=row["source"], checkpoint=which, frozen=digest(INPUT),
                              initial=fingerprint(ids), trace=trace))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","collect"))
    p.add_argument("--donor",choices=("base","instruction"))
    p.add_argument("--block",type=int, choices=(0,1))
    args=p.parse_args()
    if args.action=="freeze":
        freeze()
    else:
        collect(args.block,args.donor)
