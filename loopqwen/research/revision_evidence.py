"""Run and independently recount free judgements on actual edited Wikipedia evidence."""

import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from ..compact_entailment import MARKER, build, load
from ..evaluation.generation_eval import generate
from ..evaluation.mechanism import packed, unpack
from ..evaluation.relations import conclusion
from ..evaluation.statistics import exact_mcnemar
from ..export_runtime import load_export_tokenizer
from ..generate_cli import render
from ..io import digest, read, save
from ..ordinary_model import ordinary
from . import core
from scripts.revision_material import NAME, PANEL, REG, material


def score(tok, row, record, arm):
    text, extra = render(tok, row["body"], True)
    ids = tok.encode(text, add_special_tokens=False)
    assert ids == row["prompt_ids"] and record["prompt_tokens"] == len(ids)
    tokens = record["token_ids"]
    stops = {*extra, *tok.eos_token_ids}
    assert record["text"] == tok.decode(tokens) and not (set(tokens[:-1]) & stops)
    assert tokens[-1] in stops or len(tokens) == record["max_tokens"]
    opening = core.atom(tok, "<think>")
    closing = core.atom(tok, "</think>")
    complete = (
        tokens[-1] in stops and tokens.count(closing) == 1 and opening not in tokens
    )
    visible = tok.decode(tokens[tokens.index(closing) + 1 : -1]) if complete else ""
    sites = [(i, 0) for i in range(3, 24, 4)]
    if arm == "candidate":
        sites.extend([(11, 1), (15, 1)])
    hist = record["kv_history"]
    assert sorted((h["layer"], h["turn"]) for h in hist) == sorted(sites)
    assert all(h["offset"] == h["retained"] == len(ids) + len(tokens) - 1 for h in hist)
    answer = conclusion(visible) if complete else None
    return dict(
        complete=complete,
        visible=visible,
        answer=answer,
        correct=answer == {"SUPPORTS": "yes", "REFUTES": "no"}[row["label"]],
    )


def recount(root, rows, s):
    work = {}
    totals = {}
    release = Path(s["release"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    for arm in ("candidate", "base"):
        records = unpack(root / (arm + ".json.gz"))
        assert [r["id"] for r in rows] == [r["id"] for r in records]
        for row, r in zip(rows, records, strict=True):
            assert r["max_tokens"] == s["generation_cap"] and r["greedy"]
            assert score(tok, row, r, arm) == r["score"]
        pairs = {row["case"]: [] for row in rows}
        for row, r in zip(rows, records, strict=True):
            pairs[row["case"]].append(r["score"]["correct"])
        assert len(pairs) == s["pairs"] and all(len(v) == 2 for v in pairs.values())
        work[arm] = {k: all(v) for k, v in pairs.items()}
        first = sum(r["ttft_s"] for r in records)
        decode = sum(r["elapsed_s"] - r["ttft_s"] for r in records)
        setup = sum(r["load_s"] for r in records)
        totals[arm] = dict(
            pairs=sum(work[arm].values()),
            correct=sum(r["score"]["correct"] for r in records),
            supports=sum(
                r["score"]["correct"]
                for row, r in zip(rows, records)
                if row["label"] == "SUPPORTS"
            ),
            refutes=sum(
                r["score"]["correct"]
                for row, r in zip(rows, records)
                if row["label"] == "REFUTES"
            ),
            complete=sum(r["score"]["complete"] for r in records),
            stream_s=sum(r["request_s"] for r in records) + setup,
            first_s=sum(r["first_s"] for r in records) + setup,
            load_s=setup,
            peak_bytes=max(r["peak_bytes"] for r in records),
            prefill_tps=sum(r["prompt_tokens"] for r in records) / first,
            decode_tps=sum(len(r["token_ids"]) - 1 for r in records) / decode,
        )
    wins = sum(v and not work["base"][k] for k, v in work["candidate"].items())
    losses = sum(v and not work["candidate"][k] for k, v in work["base"].items())
    p = exact_mcnemar(wins, losses)
    c, b = totals["candidate"], totals["base"]
    gates = dict(
        opposing_pairs=wins - losses >= 5 and p <= 0.05,
        supports=c["supports"] > b["supports"],
        refutes=c["refutes"] > b["refutes"],
        completion=c["complete"] >= b["complete"],
    )
    for key in ("stream_s", "first_s", "peak_bytes"):
        gates[key] = c[key] < b[key]
    for key in ("prefill_tps", "decode_tps"):
        gates[key] = c[key] > b[key]
    return dict(
        scores=totals,
        wins=wins,
        losses=losses,
        paired_p=p,
        gates=gates,
        passed=all(gates.values()),
    )


def prerequisites(s):
    qual = Path(s["qualification"])
    assert (
        digest(qual / "checksums.json") == s["qualification_sha256"]
        and read(qual / "decision.json")["passed"]
    )
    assert (
        digest(qual / "decision.json") == read(qual / "checksums.json")["decision.json"]
    )
    registry = read("research/registry.json")
    assert registry["experiments"][NAME]["registration"] == digest(REG)
    assert registry["datasets"][str(PANEL)]["sha256"] == digest(PANEL)
    rows = material()
    assert rows == read(PANEL)
    return rows


def verify(root, d):
    s = read(REG)
    assert read(root / "registration.json") == s and d["data"] == digest(PANEL)
    rows = prerequisites(s)
    if d["status"] == "completed":
        assert all(d[k] == v for k, v in recount(root, rows, s).items())
        assert read(root / MARKER)["relative_logit_rms"] < 0.06
    else:
        assert not d["passed"] and d["reason"]


def run():
    s = read(REG)
    rows = prerequisites(s)
    root = core.OUT / NAME
    root.mkdir(parents=True, exist_ok=False)
    b = core.Budget(NAME, s["limits"])
    result = dict(data=digest(PANEL), registration=digest(REG), passed=False)
    export = Path(s["export"])
    records = {"candidate": [], "base": []}
    try:
        cert = build(Path(s["release"]), Path(s["model"]), export)
        save(root / MARKER, cert)
        b.check()
        half = len(rows) // 2
        for offset, order in (
            (0, ("candidate", "base")),
            (half, ("base", "candidate")),
        ):
            for arm in order:
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                start = perf_counter()
                model, tok = (
                    load(export) if arm == "candidate" else ordinary(s["release"])[:2]
                )
                loaded = perf_counter() - start
                assert model.span.passes == (2 if arm == "candidate" else 1)
                for i, row in enumerate(rows[offset : offset + half]):
                    start = perf_counter()
                    body, stops = render(tok, row["body"], True)
                    assert (
                        tok.encode(body, add_special_tokens=False) == row["prompt_ids"]
                    )
                    first = []

                    def progress(consumed):
                        if consumed == len(row["prompt_ids"]) and not first:
                            first.append(perf_counter() - start)
                        b.check()

                    r = generate(
                        model,
                        tok,
                        body,
                        s["generation_cap"],
                        stop_ids=stops,
                        record_kv_history=True,
                        progress=progress,
                    )
                    mx.synchronize()
                    r.update(
                        id=row["id"],
                        request_s=perf_counter() - start,
                        first_s=first[0],
                        load_s=loaded if i == 0 else 0,
                    )
                    del r["prompt"]
                    r["score"] = score(tok, row, r, arm)
                    records[arm].append(r)
                    packed(root / (arm + ".json.gz"), records[arm])
                    print(arm, len(records[arm]), row["id"], flush=True)
                del model, tok
        result.update(status="completed", **recount(root, rows, s))
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
