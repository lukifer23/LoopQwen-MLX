"""Reserve complete expert-written long-context questions without using their labels for selection."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
import shutil

from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.program_language_material import user
from scripts.table_policy_material import persist

REV = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
FILE = Path("data/research_external/longbench-v2/data.json")
OUT = Path("research/sources/human-long-context/panel.json")


def records():
    # Each context is loaded separately. The canonical array is 465 MB in UTF-8.
    decoder = json.JSONDecoder()
    buf = ""
    with FILE.open() as source:
        assert source.read(1) == "["
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            try:
                record, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = source.read(2**20)
                if not chunk:
                    raise
                buf += chunk
                continue
            yield record
            buf = buf[end:]


def prompt(w):
    return (
        w["context"]
        + "\n\n"
        + w["question"]
        + "\n"
        + "\n".join(f"{k}. {w['choice_' + k]}" for k in "ABCD")
        + "\nGive your answer as one of the four choices and explain your reasoning using the document."
    )


def run(download):
    if download and not FILE.exists():
        FILE.parent.mkdir(exist_ok=True, parents=True)
        tmp = FILE.with_suffix(".part")
        with (
            urlopen(
                f"https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/{REV}/data.json",
                timeout=90,
            ) as f,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(f, out)
        assert digest(tmp) == SHA
        tmp.rename(FILE)
    assert digest(FILE) == SHA
    domains = Counter()
    pool = []
    for r in records():
        domains[(r["domain"], r["sub_domain"])] += 1
        # Titles, lengths and record IDs only determine eligibility and order.
        if (
            r["domain"]
            in (
                "Single-Document QA",
                "Multi-Document QA",
                "Long-dialogue History Understanding",
            )
            and 180000 < len(r["context"]) < 1550000
        ):
            pool.append(
                (
                    fingerprint([781923, r["_id"]]),
                    r["_id"],
                    r["domain"],
                    r["sub_domain"],
                )
            )
    listing = sorted(pool)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    selected = []
    used = set()
    contexts = set()
    # A full scan loads one context at a time. Only the first selected member of
    # a source is retained. Independent sources matter more than more prompts.
    eligible = {i: (score, domain, sub) for score, i, domain, sub in listing}
    for wave in range(0, len(listing), 24):
        ids = {r[1] for r in listing[wave : wave + 24]}
        choices = []
        for w in records():
            if w["_id"] not in ids:
                continue
            group = w["sub_domain"]
            ctx = hashlib.sha256(w["context"].encode()).hexdigest()
            if group in used or ctx in contexts:
                continue
            question = prompt(w)
            p = tok.encode(render(tok, [user(question)], []), add_special_tokens=False)
            if not 131072 <= len(p) <= 200000:
                continue
            choices.append(
                dict(
                    id=w["_id"],
                    domain=w["domain"],
                    subdomain=group,
                    context=ctx,
                    question=fingerprint(w["question"]),
                    input_ids=fingerprint(p),
                    tokens=len(p),
                    documented_difficulty=w["difficulty"],
                    documented_length=w["length"],
                )
            )
        for w in sorted(choices, key=lambda r: eligible[r["id"]][0]):
            if w["subdomain"] in used or w["context"] in contexts:
                continue
            used.add(w["subdomain"])
            contexts.add(w["context"])
            selected.append(w)
        if len(selected) >= 4:
            break
    assert len(selected) >= 3, (domains, len(pool), selected)
    selected = selected[:4]
    persist(
        OUT,
        dict(
            revision=REV,
            payload=SHA,
            bytes=FILE.stat().st_size,
            upstream="https://huggingface.co/datasets/zai-org/LongBench-v2",
            domains={a + " / " + b: n for (a, b), n in domains.items()},
            source_records=sum(domains.values()),
            panel=selected,
            protocol="Reserve expert-authored new sources for a future release confirmation only. Rank by original ID, without reading correctness labels or model outputs. Retain an intact original context, original question and four original options. Use the standard thinking template, no answer-bearing excerpts, no truncation, no model-aware source selection. At most one question per context digest and subdomain, actual input 131072 to 200000 tokens. Record only source/question/input hashes and counts. Do not use this material for model fitting. This is input qualification only; no inference, accuracy or empty-credit release claim.",
        ),
    )
    print([(w["subdomain"], w["tokens"]) for w in selected])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    run(p.parse_args().download)
