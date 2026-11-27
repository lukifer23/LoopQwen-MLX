"""Reconstruct the frozen opposing evidence from the VitaminC publishers' original test file."""

import json
from collections import defaultdict
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

from loopqwen.export_runtime import load_export_tokenizer
from loopqwen.generate_cli import render
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

NAME = "revision-evidence-01"
REG = Path("research/experiments") / (NAME + ".json")
PANEL = Path("research/data") / (NAME + ".json")
SOURCE = Path("data/research_external/vitaminc/test.jsonl")


def material():
    s = read(REG)
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SOURCE.with_suffix(".download")
        with urlopen(s["source"], timeout=90) as inp, tmp.open("wb") as out:
            copyfileobj(inp, out)
        if digest(tmp) != s["source_sha256"]:
            raise ValueError("VitaminC download differs")
        tmp.replace(SOURCE)
    if digest(SOURCE) != s["source_sha256"]:
        raise ValueError("VitaminC test split differs")
    groups = defaultdict(list)
    with SOURCE.open() as stream:
        for r in map(json.loads, stream):
            if r["revision_type"] == "real":
                groups[
                    (r["page"], r["wiki_revision_id"], r["case_id"], r["claim"])
                ].append(r)
    release = Path(s["release"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    pages, rows = set(), []
    for key in sorted(groups, key=lambda k: fingerprint([s["seed"], "pair", k])):
        page, rev, case, claim = key
        pair = groups[key]
        if (
            page in pages
            or len(pair) != 2
            or {r["label"] for r in pair} != {"SUPPORTS", "REFUTES"}
            or len({r["evidence"] for r in pair}) != 2
            or any("\ufffd" in claim + r["evidence"] for r in pair)
        ):
            continue
        pages.add(page)
        for r in pair:
            body = f"Article excerpt:\n{r['evidence']}\n\nClaim:\n{claim}\n\nBased on this excerpt, is the claim supported? Give your final answer as yes or no."
            text, _ = render(tok, body, True)
            rows.append(
                dict(
                    id=r["unique_id"],
                    page=page,
                    revision=rev,
                    case=case,
                    claim=claim,
                    evidence=r["evidence"],
                    label=r["label"],
                    body=body,
                    prompt_ids=tok.encode(text, add_special_tokens=False),
                )
            )
        if len(pages) == s["pairs"]:
            break
    assert len(rows) == 2 * s["pairs"]
    return sorted(rows, key=lambda r: fingerprint([s["seed"], "order", r["id"]]))


if __name__ == "__main__":
    rows = material()
    if not PANEL.exists():
        save(PANEL, rows)
    assert rows == read(PANEL)
    print(len(rows), "opposing-evidence requests", digest(PANEL))
