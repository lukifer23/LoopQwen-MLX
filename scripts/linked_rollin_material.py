"""Bind real user requests and source-based teaching actions to the same deployed CSV interface."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.linked_table_material import cases, documents, render
from scripts.linked_table_sources import contains, words

TRAIN = Path("research/data/linked-rollin-training.json")
PANEL = Path("research/data/linked-rollin-panel.json")


def quotation(passage, answer):
    candidates = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", passage)
        if contains(words(s), words(answer))
    ]
    if candidates:
        return min(candidates, key=len)
    # Preserve an exact source quotation even when the answer crosses a tokenized period.
    return passage


def bind_links(tok, w, source, pages, folder):
    _, docs = documents(source, pages)
    rows = []
    for mode, v in render(w, source, pages, folder).items():
        q = v["question"]
        path = w["question_id"]
        ids = tok.encode(
            qwen_tools.render(tok, [{"role": "user", "content": q}], v["definitions"]),
            add_special_tokens=False,
        )
        evidence = v["events"][-1]["observation"]["value"]
        sentence = quotation(evidence, w["answer"])
        assert sentence in evidence and contains(words(sentence), words(w["answer"]))
        rows.append(
            {
                "id": path + "-" + mode,
                "cluster": path,
                "family": "linked_" + mode,
                "files": [path + "/" + k for k in v["files"]],
                "questions": [q],
                "initial_ids": ids,
                "gold": w["answer"],
                "page": w["link"],
                "events": v["events"],
                "quote": sentence,
                "cell": w["cell"],
                "source_url": w["table_url"],
            }
        )
    return rows, {w["question_id"] + "/" + k: v for k, v in docs.items()}


def build(*, text_disjoint=False):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    sets = {s: {"rows": [], "documents": {}} for s in ("train", "dev")}
    with TemporaryDirectory() as tmp:
        for split, w, source, pages in cases(text_disjoint=text_disjoint):
            rows, docs = bind_links(tok, w, source, pages, Path(tmp))
            sets[split]["rows"].extend(rows)
            sets[split]["documents"].update(docs)
    earlier = read("research/data/valid-return-panel.json")
    prior = earlier["rows"]
    include = set()
    for fam, n in (("inline", 2), ("wheel", 4), ("agenda", 4)):
        ids = sorted(
            {w["cluster"] for w in prior if w["family"] == fam},
            key=lambda k: fingerprint([67129, k]),
        )
        include.update(ids[:n])
    selected = [
        w for w in prior if w["cluster"] in include or w["family"] == "retention"
    ]
    assert len(selected) == 36
    p = sets["dev"]
    p["rows"].extend(selected)
    p["documents"].update(
        {
            w["document"]: earlier["documents"][w["document"]]
            for w in selected
            if w["document"]
        }
    )
    assert (len(sets["train"]["rows"]), len(p["rows"])) == (256, 84)
    train = (
        TRAIN.with_name("linked-rollin-text-training.json") if text_disjoint else TRAIN
    )
    for dest, data in ((train, sets["train"]), (PANEL, p)):
        if dest.exists():
            assert read(dest) == data
        else:
            save(dest, data)
    print(
        "256 training views, 48 fresh open-answer requests and 36 fixed earlier retention requests."
    )


if __name__ == "__main__":
    build()
