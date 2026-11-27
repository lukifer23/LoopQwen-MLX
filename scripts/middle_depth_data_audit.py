"""Verify source identities, prompt separation and paired exposure in the native study."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.middle_depth_budget import BASE, ROOT
from scripts.middle_depth_development import (
    FROZEN,
    SOURCE,
    WORLDS,
    instance,
    prompt_ids,
)
from scripts.middle_depth_material import ROWS


def check():
    data = unpack(ROWS)
    cfg = read(ROWS.with_name("inventory.json"))
    assert digest(ROWS) == cfg["training"]
    for f, sha in cfg["source_hashes"].items():
        if digest(Path(f)) != sha:
            raise ValueError("Changed training source " + f)
    inputs = {fingerprint(r["p"]) for r in data}
    tok = tokenizer({"checkpoint": str(BASE)})
    frozen = read(FROZEN)
    prompts = set()
    questions = []
    with TemporaryDirectory() as tmp:
        for r in read(SOURCE):
            for world in WORLDS:
                text, tools = instance(r, world, Path(tmp))
                ids = prompt_ids(tok, [dict(role="user", content=text)], tools)
                sha = fingerprint(ids)
                assert sha == frozen["ids"][r["id"] + "/" + world] and sha not in inputs
                prompts.add(sha)
                questions.append((r["id"], r["question"]))
    decoded = [tok.decode(r["p"]).casefold() for r in data]
    clipped = [
        (src, i)
        for src, q in set(questions)
        for i, s in enumerate(decoded)
        if q.casefold() in s
    ]
    train_books = read(Path("research/diagnostics/middle-depth/long-fit.json"))["rows"]
    dev_books = read(Path("research/diagnostics/middle-depth/document-openings.json"))
    assert not {r["source_sha"] for r in train_books.values()} & {
        d["source"] for d in dev_books.values()
    }
    orig_ids = [r["id"] for r in data]
    assert len(set(orig_ids)) == len(orig_ids)
    report = dict(
        training=digest(ROWS),
        inventory=digest(ROWS.with_name("inventory.json")),
        families=dict(Counter(r["family"] for r in data)),
        source_groups={
            f: len({r["cluster"] for r in data if r["family"] == f})
            for f in sorted({r["family"] for r in data})
        },
        train_prompts=len(inputs),
        unique_short_dev_prompts=len(prompts),
        short_worlds=len(questions),
        shared_exact_openings=len(inputs & prompts),
        whole_held_question_text_hits=clipped,
        whole_training_and_dev_document_overlap=0,
        matched_order=fingerprint(orig_ids * 4),
        fit_rows=len(orig_ids) * 4,
    )
    save(ROOT / "data-audit.json", report)
    if clipped:
        raise ValueError("Held question wording occurs verbatim in training")


if __name__ == "__main__":
    check()
