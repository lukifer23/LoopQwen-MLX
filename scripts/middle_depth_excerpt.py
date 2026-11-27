"""Diagnose evidence use at two natural excerpt lengths before full-book interpretation."""

import argparse
from pathlib import Path

from loopqwen.calculator_tool import tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import MiddleAssistant, donor
from loopqwen.qwen_tools import declarations, render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_documents import SPECS

SPEC = SPECS / "excerpt-probes.json"


def inputs(tok):
    for name, first in (
        ("book", "his canary bird, in its\ngilt cage"),
        ("anne", "I was eleven last March"),
    ):
        spec = read(SPECS / f"{name}-dev.json")
        body = Path(spec["source"]).read_text()
        early = body.index(first)
        late = body.index(spec["original"])
        tools = [tool()] if name == "anne" else []
        for length, radius in (("short", 1250), ("medium", 18000)):
            ranges = [
                (max(0, i - radius), min(len(body), i + radius)) for i in (early, late)
            ]
            passages = [body[a:b] for a, b in ranges]
            assert passages[0].count(first) == passages[1].count(spec["original"]) == 1
            if name == "book":
                assert "handcuff their wrists together" in passages[1]
            for world in ("original", "edited"):
                tail = (
                    passages[1]
                    if world == "original"
                    else passages[1].replace(spec["original"], spec["replacement"])
                )
                q = passages[0] + "\n\n[…]\n\n" + tail + "\n\n" + spec["question"]
                ids = tok.encode(
                    render(tok, [dict(role="user", content=q)], declarations(tools)),
                    add_special_tokens=False,
                )
                yield (
                    name,
                    length,
                    world,
                    q,
                    tools,
                    dict(
                        input=fingerprint(ids),
                        positions=len(ids),
                        source=digest(Path(spec["source"])),
                        character_ranges=ranges,
                    ),
                )


def freeze():
    rows = {
        "/".join((n, l, w)): item
        for n, l, w, q, t, item in inputs(tokenizer({"checkpoint": str(BASE)}))
    }
    save(
        SPEC,
        dict(
            rows=rows,
            tokens=512,
            protocol="Two unannotated contiguous excerpts from each complete development book, separated by an omission mark. Both relevant passages are present. Use the identical book question and decision rubric. This is an evidence-availability mechanism probe; never count excerpts toward full-length accuracy.",
        ),
    )


def evaluate(arm, policy, stage, block):
    cfg = read(SPEC)
    with worker("short", "excerpt/" + arm + "/" + str(block)) as guard:
        if policy:
            a = MiddleAssistant(BASE, policy)
            m, tok = a._tool_model, a.tokenizer
            identity = digest(policy / "adapters.safetensors")
        else:
            text, tok = donor(BASE, "base")
            m = LoopedModel(text, Span(12, 4, 1))
            m.eval()
            identity = "untouched-base"
        for i, (n, l, w, q, tools, meta) in enumerate(inputs(tok)):
            if i % 2 != block:
                continue
            key = "/".join((n, l, w))
            expected = cfg["rows"][key]
            assert fingerprint(meta) == fingerprint(expected)
            out = ROOT / "excerpts" / stage / arm / n / l / (w + ".json.gz")
            if out.exists():
                assert unpack(out)["weights"] == identity
                continue
            tr = run(
                m,
                tok,
                q,
                tools,
                tool_format="qwen",
                max_tokens=cfg["tokens"],
                max_turns=3,
                record_kv_history=True,
                prefill_cache_state=True,
                progress=guard,
            )
            assert fingerprint(tr["turns"][0]["prompt_ids"]) == meta["input"]
            packed(
                out,
                dict(
                    weights=identity,
                    trace=tr,
                    registration=digest(SPEC),
                    source=n,
                    length=l,
                    world=w,
                    meta=meta,
                ),
            )


def review(stage, aggregate):
    dest = ROOT / "excerpt-review" / stage
    items = {}
    rows = []
    for path in sorted((ROOT / "excerpts" / stage).glob("*/*/*/*.json.gz")):
        arm, book, length, _ = path.relative_to(ROOT / "excerpts" / stage).parts
        raw = unpack(path)
        world = raw["world"]
        tr = raw["trace"]
        ref = read(SPECS / f"{book}-dev.json")["rubric_" + world]
        item = dict(source=book, edition=world, reference=ref, answer=tr["answer"])
        key = fingerprint(item)
        if tr["status"] == "answered":
            items[key] = item
        rows.append(
            dict(
                arm=arm,
                book=book,
                length=length,
                world=world,
                positions=raw["meta"]["positions"],
                status=tr["status"],
                seconds=tr["seconds"],
                key=key,
                file=str(path.relative_to(ROOT)),
                sha=digest(path),
            )
        )
    save(dest / "blinded.json", {k: items[k] for k in sorted(items)})
    if aggregate:
        rates = read(dest / "grades.json") if items else {}
        assert set(rates) == set(items)
        for r in rows:
            r["grade"] = (
                rates[r["key"]]
                if r["status"] == "answered"
                else dict(correct=False, first_error=r["status"], unsupported=None)
            )
        save(dest / "scored.json", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "run", "blind", "aggregate"))
    p.add_argument("--arm", choices=("base", "loop", "ordinary"))
    p.add_argument("--policy", type=Path)
    p.add_argument("--stage", default="terminal")
    p.add_argument("--block", type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    if a.action in ("blind", "aggregate"):
        review(a.stage, a.action == "aggregate")
    elif a.action == "freeze":
        freeze()
    else:
        evaluate(a.arm, a.policy, a.stage, a.block)
