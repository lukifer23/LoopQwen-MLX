"""Free generation on both versions of the complete McTeague e-book."""
import argparse
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from loopqwen.evaluation.mechanism import packed
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import MiddleAssistant, architecture, donor
from loopqwen.qwen_tools import render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker

BOOK = Path("research/diagnostics/middle-depth/book-dev.json")
FROZEN = BOOK.with_name("book-inputs.json")


def opening(edited):
    spec = read(BOOK)
    book = Path(spec["source"]).read_text()
    assert book.count(spec["original"]) == 1
    if edited:
        book = book.replace(spec["original"], spec["replacement"])
    return book + "\n\n" + spec["question"]


def freeze():
    spec = read(BOOK)
    tok = tokenizer({"checkpoint":str(BASE)})
    rows = []
    for changed in (False,True):
        ids = tok.encode(render(tok,[dict(role="user",content=opening(changed))],[]),add_special_tokens=False)
        rows.append(dict(changed=changed, length=len(ids), input=fingerprint(ids)))
    save(FROZEN, dict(source=digest(Path(spec["source"])), spec=digest(BOOK), worlds=rows))


def measure(args):
    cfg = read(FROZEN)
    assert cfg["spec"] == digest(BOOK) and cfg["source"] == digest(Path(read(BOOK)["source"]))
    name = args.policy.parent.name+"-"+args.policy.name if args.policy else args.arm
    out = ROOT/"book"/f"{name}-{args.world}-{args.tokens}.json.gz"
    if out.exists():
        raise FileExistsError(out)
    with worker("long",str(out)) as guard:
        if args.policy:
            a=MiddleAssistant(BASE,args.policy)
            model,tok=a._tool_model,a.tokenizer
            weights=read(args.policy/"registration.json")["weights"]
        else:
            text,tok=donor(BASE,args.donor)
            model=architecture(text,2,4619) if args.arm=="initial-loop" else LoopedModel(text,Span(12,4,1))
            model.eval()
            weights=args.donor
        scratch=Path("outputs/middle-depth/staging")
        scratch.mkdir(parents=True,exist_ok=True)
        mx.reset_peak_memory()
        trace=run(model,tok,opening(args.world=="edited"),[],max_tokens=args.tokens,max_turns=1,
                  staged_context_dir=scratch,tool_format="qwen",progress=guard,record_kv_history=True)
        first=trace["turns"][0]
        ids=first["prompt_ids"]
        expected=cfg["worlds"][args.world=="edited"]
        assert len(ids)==expected["length"] and fingerprint(ids)==expected["input"]
        consumed=len(ids)+len(first["token_ids"])-1
        assert len(first["kv_history"])==sum(not model.text.layers[i].is_linear for i,p in model.sites)
        assert all(h["offset"]==h["retained"]==consumed for h in first["kv_history"])
        packed(out,dict(frozen=digest(FROZEN), weights=weights, sites=model.sites, trace=trace))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run"))
    p.add_argument("--donor",choices=("instruction","base"),default="base")
    p.add_argument("--arm",choices=("base","initial-loop","instruction"))
    p.add_argument("--tokens",type=int,choices=(128,512),default=128)
    p.add_argument("--world",choices=("original","edited"),default="original")
    p.add_argument("--policy",type=Path)
    args=p.parse_args()
    freeze() if args.action=="freeze" else measure(args)
