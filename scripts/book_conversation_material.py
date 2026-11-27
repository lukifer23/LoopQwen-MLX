"""Freeze two complete books, independent reader questions, and native function prompts."""

import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen import file_tools, qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

SOURCES = [
    (
        "The House of the Seven Gables",
        "The-House-of-the-Seven-Gables_77/e15cebd7ca877c137ca39692bcb09f78e74adcbb/77.txt",
        "seven-gables.txt",
        "How long did Clifford serve time for man slaughter ?",
        ["30 years.", "thirty years"],
        30,
    ),
    (
        "He Walked Around the Horses",
        "He-Walked-Around-the-Horses_18807/00cd14328626ff6546440a7d93735ee5b4ee4ee3/18807.txt",
        "horses.txt",
        "How old was Bathurst when he disappeared?",
        ["25", "25"],
        25,
    ),
]
DATA = Path("research/data/book-conversation-panel.json")
PUBLIC = Path("data/research_external/narrativeqa")


def source(item):
    path = Path(item["file"])
    if not path.exists():
        data = urlopen(item["url"], timeout=60).read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("The original book bytes differ")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert digest(path) == item["sha256"]
    return path


def messages(item):
    path = source(item)
    text = path.read_bytes().decode("utf-8-sig")
    return [
        f"Here is the complete text of {path}.\n\n{text}\n\n{item['question']} Give the number of years.",
        f"Please use the file tool to check the actual size in bytes of {path}.",
        "How many lines does that same file contain? Please check with the file tool.",
    ]


def main():
    tokenizer = load_tokenizer(local_snapshot())
    docs = list(csv.DictReader((PUBLIC / "documents.csv").open()))
    annotations = list(csv.DictReader((PUBLIC / "qaps.csv").open()))
    items = []
    for title, archived, filename, question, answers, number in SOURCES:
        repo = "https://raw.githubusercontent.com/GITenberg/" + archived
        raw = urlopen(repo, timeout=60).read()
        d = next(
            r
            for r in docs
            if r["wiki_title"].casefold() == title.casefold() and r["set"] == "test"
        )
        q = next(
            q
            for q in annotations
            if q["document_id"] == d["document_id"] and q["question"] == question
        )
        assert [q["answer1"], q["answer2"]] == answers
        item = {
            "title": title,
            "document_id": d["document_id"],
            "question": question,
            "annotations": answers,
            "reference": number,
            "url": repo,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "file": str(PUBLIC / "field-books" / filename),
            "size": len(raw),
            "lines": len(raw.splitlines()),
        }
        path = source(item)
        assert path.read_bytes() == raw
        definitions = qwen_tools.declarations(file_tools.tools([path]))
        rendered = qwen_tools.render(
            tokenizer, [{"role": "user", "content": messages(item)[0]}], definitions
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        item.update(initial_ids=ids, initial_sha256=fingerprint(ids))
        print(title, len(ids), len(raw), len(raw.splitlines()))
        items.append(item)
    assert (
        131072 <= len(items[0]["initial_ids"]) < 190000
        and len(items[1]["initial_ids"]) < 40000
    )
    out = {
        "items": items,
        "annotation_sha256": digest(PUBLIC / "qaps.csv"),
        "documents_sha256": digest(PUBLIC / "documents.csv"),
    }
    if DATA.exists():
        assert read(DATA) == out
    else:
        save(DATA, out)


if __name__ == "__main__":
    main()
