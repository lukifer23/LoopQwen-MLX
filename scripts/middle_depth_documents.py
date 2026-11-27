"""Persistent exact-prefix measurements for natural long documents and tool follow-ups."""

import argparse
from pathlib import Path
from time import monotonic

import mlx.core as mx

from loopqwen.calculator_tool import tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import MiddleAssistant, architecture, donor
from loopqwen.qwen_tools import declarations, render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.saved_document import SavedDocument
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, worker

SPECS = Path("research/diagnostics/middle-depth")
FROZEN = SPECS / "document-openings.json"


def worlds(name, tok):
    spec = read(SPECS / f"{name}-dev.json")
    source = Path(spec["source"])
    book = source.read_text()
    assert book.count(spec["original"]) == 1
    tools = [tool()] if name == "anne" else []
    w = {}
    for world in ("original", "edited"):
        body = (
            book.replace(spec["original"], spec["replacement"])
            if world == "edited"
            else book
        )
        question = body + "\n\n" + spec["question"]
        p = tok.encode(
            render(tok, [dict(role="user", content=question)], declarations(tools)),
            add_special_tokens=False,
        )
        w[world] = (question, p)
    length = next(
        i for i, (a, b) in enumerate(zip(w["original"][1], w["edited"][1])) if a != b
    )
    # Align to public prefill blocks before branching. The original and edited
    # questions keep every subsequent input token and the appended chat template.
    length = length // 128 * 128
    ids = w["original"][1][:length]
    assert ids == w["edited"][1][:length] and length > 131072
    return spec, tools, w, ids


def freeze():
    tok = tokenizer({"checkpoint": str(BASE)})
    results = {}
    for name in ("book", "anne"):
        spec, t, w, ids = worlds(name, tok)
        results[name] = dict(
            spec=digest(SPECS / f"{name}-dev.json"),
            source=digest(Path(spec["source"])),
            common=len(ids),
            common_sha=fingerprint(ids),
            worlds={
                k: dict(input=fingerprint(p), length=len(p)) for k, (q, p) in w.items()
            },
        )
    save(FROZEN, results)


def measure(args):
    cfg = read(FROZEN)[args.book]
    with worker(
        "long", f"{args.book}/{args.arm}/{args.world}/{args.tokens}/{args.stage}"
    ) as guard:
        if args.policy:
            assist = MiddleAssistant(BASE, args.policy)
            model, tok = assist._tool_model, assist.tokenizer
            weights = read(args.policy / "registration.json")["weights"]
        else:
            text, tok = donor(BASE, "base")
            model = (
                architecture(text, 2, 4619)
                if args.arm == "loop"
                else LoopedModel(text, Span(12, 4, 1))
            )
            model.eval()
            weights = "base-dc7cdfe"
        spec, t, w, ids = worlds(args.book, tok)
        assert cfg["spec"] == digest(SPECS / f"{args.book}-dev.json") and cfg[
            "source"
        ] == digest(Path(spec["source"]))
        assert cfg["common_sha"] == fingerprint(ids)
        q, full = w[args.world]
        out = ROOT / "documents" / args.stage / args.book / args.arm / str(args.tokens)
        ident = fingerprint(
            [weights, model.sites, cfg["common_sha"], "saved-prefix-v1"]
        )
        directory = Path("outputs/middle-depth/documents") / ident
        prep = monotonic()
        prefix = SavedDocument(model, ids, directory, ident, guard)
        prefill_seconds = monotonic() - prep

        def take(kind, question, history):
            path = out / (args.world + "-" + kind + ".json.gz")
            if path.exists():
                record = unpack(path)
                assert record["identity"] == ident and record["frozen"] == digest(
                    FROZEN
                )
                return record["history"]
            times = []
            start = monotonic()

            def timer(n):
                guard()
                times.append((n, monotonic() - start))

            trace = run(
                model,
                tok,
                question,
                t,
                max_tokens=args.tokens,
                max_turns=3,
                document_prefix=prefix,
                tool_format="qwen",
                record_kv_history=True,
                progress=timer,
                _messages=history,
            )
            if kind == "first":
                assert trace["turns"][0]["prompt_ids"] == full
            for turn in trace["turns"]:
                length = len(turn["prompt_ids"]) + len(turn["token_ids"]) - 1
                assert len(turn["kv_history"]) == sum(
                    not model.text.layers[i].is_linear for i, p in model.sites
                )
                assert all(
                    k["offset"] == k["retained"] == length for k in turn["kv_history"]
                )
            packed(
                path,
                dict(
                    identity=ident,
                    weights=weights,
                    frozen=digest(FROZEN),
                    timing=times,
                    trace=trace,
                    source=cfg["source"],
                    history=history,
                    sites=model.sites,
                    prefix_seconds=prefill_seconds,
                    cold_build_seconds=prefix.receipt["seconds"],
                    scratch_bytes=prefix.disk.scratch_bytes,
                ),
            )
            return history

        history = take("first", q, [])
        if args.followups:
            for kind, h in (
                ("own", history),
                (
                    "shared",
                    [
                        dict(role="user", content=q),
                        dict(role="assistant", content=spec["rubric_" + args.world]),
                    ],
                ),
            ):
                take(kind, spec["followup"], list(h))
            take(
                "unanswerable",
                spec["unanswerable"],
                [
                    dict(role="user", content=q),
                    dict(role="assistant", content=spec["rubric_" + args.world]),
                ],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "run"))
    p.add_argument("--book", choices=("book", "anne"), default="book")
    p.add_argument("--arm", choices=("base", "loop", "ordinary"), default="base")
    p.add_argument("--stage", default="initial")
    p.add_argument("--world", choices=("original", "edited"), default="original")
    p.add_argument("--tokens", choices=(128, 512, 4096), type=int, default=128)
    p.add_argument("--followups", action="store_true")
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    freeze() if a.action == "freeze" else measure(a)
