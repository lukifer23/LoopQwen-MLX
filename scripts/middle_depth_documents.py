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
from loopqwen.middle_depth import INSTRUCTION_REV, MiddleAssistant, architecture, donor
from loopqwen.qwen_tools import declarations, render
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.resident_document import ResidentDocument
from loopqwen.tool_session import run
from scripts.middle_depth_budget import BASE, ROOT, DevelopmentLimit, worker
from scripts.middle_depth_retention import saved

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
            render(tok, [{"role": "user", "content": question}], declarations(tools)),
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
        spec, _t, w, ids = worlds(name, tok)
        results[name] = {
            "spec": digest(SPECS / f"{name}-dev.json"),
            "source": digest(Path(spec["source"])),
            "common": len(ids),
            "common_sha": fingerprint(ids),
            "worlds": {
                k: {"input": fingerprint(p), "length": len(p)}
                for k, (q, p) in w.items()
            },
        }
    save(FROZEN, results)


EXTRA = SPECS / "document-extra-openings.json"


IRRELEVANT = SPECS / "document-irrelevant-edits.json"
STABLE = SPECS / "document-irrelevant-openings.json"


def irrelevant_opening(book, tok, original, tools, common):
    before, after = read(IRRELEVANT)[book]
    assert original.count(before) == 1
    q = original.replace(before, after)
    p = tok.encode(
        render(tok, [{"role": "user", "content": q}], declarations(tools)),
        add_special_tokens=False,
    )
    assert p[: len(common)] == common
    return q, p


def freeze_irrelevant():
    tok = tokenizer({"checkpoint": str(BASE)})
    rows = {}
    for name in ("book", "anne"):
        spec, tools, w, common = worlds(name, tok)
        _q, p = irrelevant_opening(name, tok, w["original"][0], tools, common)
        rows[name] = {
            "input": fingerprint(p),
            "length": len(p),
            "reference": spec["rubric_original"],
        }
    save(
        STABLE, {"edits": digest(IRRELEVANT), "original": digest(FROZEN), "rows": rows}
    )


def extra_openings():
    tok = tokenizer({"checkpoint": str(BASE)})
    spec, tools, w, ids = worlds("anne", tok)
    rows = {}
    for world, (q, p) in w.items():
        for kind, msgs in (
            (
                "shared",
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": spec["rubric_" + world]},
                    {"role": "user", "content": spec["followup"]},
                ],
            ),
            (
                "unanswerable",
                [
                    {
                        "role": "user",
                        "content": q.removesuffix(spec["question"])
                        + spec["unanswerable"],
                    }
                ],
            ),
        ):
            tokens = tok.encode(
                render(tok, msgs, declarations(tools)), add_special_tokens=False
            )
            assert tokens[: len(ids)] == ids
            rows[world + "/" + kind] = {
                "input": fingerprint(tokens),
                "length": len(tokens),
            }
    save(
        EXTRA,
        {
            "first": digest(FROZEN),
            "rows": rows,
            "protocol": "Primary book inputs contain the complete published or edited document. The unanswerable asks for an undocumented clock minute without a supplied first answer. On separate follow-ups, own history retains actual model exchanges and shared history fixes a common supported first response.",
        },
    )


def qualified_resident(model, prefix, full, ident, output_tokens, guard):
    if not read(ROOT / "resident-qualification.json")["passed"]:
        raise ValueError("Long output and callback equivalence prerequisite failed")
    audit = ROOT / "resident-checks" / (ident + ".json")
    live = ResidentDocument(prefix, output_tokens, guard)
    if not audit.exists():
        start = len(prefix.ids)
        opening = full[: start + 128]
        probes = []
        for reader in (prefix, live):
            child = reader.branch(model, opening)
            ids = mx.array([opening[start:]])
            scores = []
            for _ in range(2):
                guard()
                logits = child.decode(ids)[:, -1]
                mx.eval(logits)
                scores.append(logits)
                ids = mx.argmax(logits, axis=-1)[:, None]
            probes.append(scores)
            del child
        errors = [mx.max(mx.abs(a - b)).item() for a, b in zip(*probes, strict=True)]
        save(
            audit,
            {
                "passed": max(errors) == 0,
                "identity": ident,
                "sites": model.sites,
                "input": fingerprint(opening),
                "positions": len(opening),
                "maximum_logit_error": errors,
            },
        )
    if not read(audit)["passed"]:
        raise ValueError("This exact deployed prefix failed resident logit parity")
    return live


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
            which = "instruction" if args.arm == "instruction" else "base"
            text, tok = donor(BASE, which)
            model = (
                architecture(text, 2, 4619)
                if args.arm == "loop"
                else LoopedModel(text, Span(12, 4, 1))
            )
            model.eval()
            weights = (
                "base-dc7cdfe" if which == "base" else "instruction-" + INSTRUCTION_REV
            )
        spec, t, w, ids = worlds(args.book, tok)
        assert cfg["spec"] == digest(SPECS / f"{args.book}-dev.json") and cfg[
            "source"
        ] == digest(Path(spec["source"]))
        assert cfg["common_sha"] == fingerprint(ids)
        if args.world == "irrelevant":
            assert not args.followups
            q, full = irrelevant_opening(args.book, tok, w["original"][0], t, ids)
            fixed = read(STABLE)
            assert fixed["edits"] == digest(IRRELEVANT) and fixed["original"] == digest(
                FROZEN
            )
            assert fixed["rows"][args.book]["input"] == fingerprint(full)
        else:
            q, full = w[args.world]
        out = ROOT / "documents" / args.stage / args.book / args.arm / str(args.tokens)
        ident = fingerprint(
            [weights, model.sites, cfg["common_sha"], "saved-prefix-v1"]
        )
        directory = Path("outputs/middle-depth/documents") / ident
        prep = monotonic()
        import sys

        prefix = saved(
            model,
            ids,
            directory,
            ident,
            guard,
            [".venv/bin/python", "-m", "scripts.middle_depth_documents", *sys.argv[1:]],
        )
        prefill_seconds = monotonic() - prep
        operating_point = args.cache_mode
        pressure_receipt = ROOT / "resident-checks" / (ident + "-pressure.json")
        if operating_point == "resident" and pressure_receipt.exists():
            fallback = read(pressure_receipt)
            if fallback["identity"] != ident or fallback["weights"] != weights:
                raise ValueError(
                    "Resource receipt belongs to a different deployed prefix"
                )
            operating_point = "staged"

        def record_pressure(error):

            save(
                pressure_receipt,
                {
                    "identity": ident,
                    "weights": weights,
                    "while_executing": "resident",
                    "error": str(error),
                    "sample": error.snapshot,
                    "remaining_worker_seconds": error.remaining_seconds,
                    "fallback": "exact staged histories on next invocation",
                },
            )

        reader = prefix
        runtime_proof = None
        if operating_point == "staged" and pressure_receipt.is_file():
            from scripts.middle_depth_staged_verify import (
                PROOF,
                alternate,
                implementation,
            )

            if PROOF.is_file():
                verified = read(PROOF)
                identity = verified["identity"]
                if (
                    verified["passed"]
                    and ident == identity["prefix"]
                    and identity["implementation"] == implementation()
                ):
                    reader = alternate(model, prefix)
                    operating_point = "staged-visits"
                    runtime_proof = digest(PROOF)
        if operating_point == "resident":
            try:
                reader = qualified_resident(
                    model, prefix, full, ident, args.tokens, guard
                )
            except DevelopmentLimit as error:
                if error.snapshot["pressure"] != 1:
                    record_pressure(error)
                raise

        def take(kind, question, history):
            path = out / (args.world + "-" + kind + ".json.gz")
            if path.exists():
                record = unpack(path)
                assert record["identity"] == ident and record["frozen"] == digest(
                    FROZEN
                )
                assert record.get("cache_mode", "staged") in (
                    args.cache_mode,
                    operating_point,
                )
                return record["history"]
            times = []
            start = monotonic()

            def timer(n):
                guard()
                times.append((n, monotonic() - start))

            try:
                trace = run(
                    model,
                    tok,
                    question,
                    t,
                    max_tokens=args.tokens,
                    max_turns=3,
                    document_prefix=reader,
                    tool_format="qwen",
                    record_kv_history=True,
                    progress=timer,
                    _messages=history,
                )
            except DevelopmentLimit as error:
                if operating_point == "resident" and error.snapshot["pressure"] != 1:
                    record_pressure(error)
                raise
            if kind == "first":
                assert trace["turns"][0]["prompt_ids"] == full
            elif kind in ("shared", "unanswerable"):
                extra = read(EXTRA)
                assert extra["first"] == digest(FROZEN)
                ids_used = trace["turns"][0]["prompt_ids"]
                assert extra["rows"][args.world + "/" + kind] == {
                    "input": fingerprint(ids_used),
                    "length": len(ids_used),
                }
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
                {
                    "cache_mode": operating_point,
                    "requested_cache_mode": args.cache_mode,
                    "staged_visit_qualification": runtime_proof,
                    "fallback_receipt": digest(pressure_receipt)
                    if operating_point != args.cache_mode
                    else None,
                    "identity": ident,
                    "weights": weights,
                    "frozen": digest(FROZEN),
                    "timing": times,
                    "trace": trace,
                    "source": cfg["source"],
                    "history": history,
                    "sites": model.sites,
                    "prefix_seconds": prefill_seconds,
                    "cold_build_seconds": prefix.cold_seconds,
                    "cold_time_basis": prefix.cold_time_basis,
                    "scratch_bytes": prefix.disk.scratch_bytes,
                },
            )
            return history

        history = take("first", q, [])
        if args.followups:
            for kind, h in (
                ("own", history),
                (
                    "shared",
                    [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": spec["rubric_" + args.world]},
                    ],
                ),
            ):
                take(kind, spec["followup"], list(h))
            take(
                "unanswerable",
                q.removesuffix(spec["question"]) + spec["unanswerable"],
                [],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "extra", "irrelevant", "run"))
    p.add_argument("--book", choices=("book", "anne"), default="book")
    p.add_argument(
        "--arm", choices=("base", "instruction", "loop", "ordinary"), default="base"
    )
    p.add_argument("--stage", default="initial")
    p.add_argument(
        "--world", choices=("original", "edited", "irrelevant"), default="original"
    )
    p.add_argument("--tokens", choices=(128, 512, 4096), type=int, default=128)
    p.add_argument("--cache-mode", choices=("resident", "staged"), default="staged")
    p.add_argument("--followups", action="store_true")
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    if a.action == "irrelevant":
        freeze_irrelevant()
    elif a.action == "extra":
        extra_openings()
    elif a.action == "freeze":
        freeze()
    else:
        measure(a)
