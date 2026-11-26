"""Rebuild file-search material from two pinned publisher test releases."""

import hashlib
import json
import random
import re
import subprocess
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen.document_tools import documents
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt

if __package__:
    from .passage_copy_material import SOURCES, normalized
else:
    from passage_copy_material import SOURCES, normalized

ROOT = Path("data/research_external")
QASPER = (
    "qasper-test-and-evaluator-v0.3.tgz",
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
)
DATA = Path("research/data/document-search-field-01.json")


def source(meta):
    name, url, expected = meta
    path = ROOT / name
    if not path.exists():
        payload = urlopen(url, timeout=90).read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Source download differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if digest(path) != expected:
        raise ValueError("Source file differs")
    return path


def inventories():
    wiki = {a["title"]: a for a in read(source(SOURCES["SQuAD", "dev"]))["data"]}
    with tarfile.open(source(QASPER)) as tar:
        papers = json.load(tar.extractfile("qasper-test-v0.3.json"))
    return {"SQuAD": wiki, "Qasper": papers}


def article(family, a):
    if family == "SQuAD":
        return "\n\n".join(
            [a["title"].replace("_", " "), *(p["context"] for p in a["paragraphs"])]
        )
    paragraphs = [a["title"], a["abstract"]]
    for section in a["full_text"]:
        paragraphs.extend([section["section_name"] or "", *section["paragraphs"]])
    return "\n\n".join(paragraphs)


def questions(family, a):
    if family == "SQuAD":
        for p in a["paragraphs"]:
            for q in p["qas"]:
                yield (
                    q["id"],
                    q["question"],
                    [(ans["text"], p["context"]) for ans in q["answers"]],
                )
    else:
        for q in a["qas"]:
            votes = []
            for ann in q["answers"]:
                ans = ann["answer"]
                if not ans["unanswerable"] and len(ans["extractive_spans"]) == 1:
                    ref = ans["extractive_spans"][0].strip()
                    evidence = next(
                        (e for e in ans["evidence"] if ref and ref in e), None
                    )
                    if evidence:
                        votes.append((ref, evidence))
            yield q["question_id"], q["question"].strip(), votes


@contextmanager
def open_panel(mat):
    bank = inventories()
    with TemporaryDirectory() as tmp:
        paths = []
        for doc in mat["documents"]:
            if doc["source"] == "Project guides":
                payload = subprocess.check_output(
                    ["git", "show", doc["revision"] + ":" + doc["key"]]
                )
            else:
                payload = article(
                    doc["source"], bank[doc["source"]][doc["key"]]
                ).encode()
            path = Path(tmp) / doc["document"]
            path.write_bytes(payload)
            assert digest(path) == doc["sha256"]
            paths.append(path)
        with documents(paths) as tools:
            yield tools


def build(
    path=DATA, seed=7145981, previous_documents=(), extra_documents=(), extra_rows=()
):
    rng = random.Random(seed)
    bank = inventories()
    tok = load_tokenizer(local_snapshot())
    material = {
        "sources": {"SQuAD": SOURCES["SQuAD", "dev"], "Qasper": QASPER},
        "documents": [],
        "rows": [],
    }
    excluded = {
        normalized(w["title"])
        for w in read("research/data/passage-copy-train.json")["rows"]
    }
    previous = set(read("research/data/natural-context-training.json")["ids"])
    old_passages = set()
    for w in read("research/data/passage-copy-dev.json")["rows"]:
        previous.add(w["id"])
        old_passages.add(normalized(w["passage"]))
    for name in ("prose-reading-01", "natural-context-01"):
        for w in read("research/data/" + name + ".json"):
            previous.add(w["id"])
            old_passages.add(
                normalized(
                    bank["SQuAD"][w["title"]]["paragraphs"][w["source_paragraph"]][
                        "context"
                    ]
                )
            )
    for family, docs in bank.items():
        keys = sorted(docs)
        rng.shuffle(keys)
        count = 0
        for key in keys:
            a = docs[key]
            if (family, key) in previous_documents:
                continue
            if (
                family == "SQuAD"
                and normalized(a["title"].replace("_", " ")) in excluded
            ):
                continue
            body = article(family, a)
            pool = list(questions(family, a))
            rng.shuffle(pool)
            for qid, q, votes in pool:
                if qid in previous or (
                    family == "SQuAD"
                    and any(normalized(e) in old_passages for _, e in votes)
                ):
                    continue
                counts = Counter(normalized(ans) for ans, _ in votes)
                if not counts:
                    continue
                best, n = counts.most_common(1)[0]
                if not 1 <= len(best.split()) <= 8 or n < 2 or n * 2 <= len(votes):
                    continue
                spans = set()
                refs = set()
                for ans, e in votes:
                    if normalized(ans) != best or e not in body:
                        continue
                    for p in re.finditer(re.escape(e), body):
                        for m in re.finditer(re.escape(ans), e):
                            spans.add((p.start() + m.start(), p.start() + m.end()))
                            refs.add(ans.strip())
                if not spans:
                    continue
                filename = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-") + ".txt"
                user = f"Using the supplied file {filename}, answer with the short answer from the text. {q}"
                material["documents"].append(
                    {
                        "source": family,
                        "key": key,
                        "document": filename,
                        "title": a["title"],
                        "sha256": hashlib.sha256(body.encode()).hexdigest(),
                        "characters": len(body),
                    }
                )
                material["rows"].append(
                    {
                        "id": qid,
                        "source": family,
                        "document": filename,
                        "question": q,
                        "request": user,
                        "refs": sorted(refs),
                        "answer_ranges": sorted(spans),
                    }
                )
                count += 1
                break
            if count == 16:
                break
        assert count == 16, (family, count)
    material["documents"].extend(extra_documents)
    material["rows"].extend(extra_rows)
    rng.shuffle(material["rows"])
    with open_panel(material) as tools:
        for w in material["rows"]:
            w["initial_ids"] = tok.encode(
                prompt(tok, initial_messages(w["request"], tools)),
                add_special_tokens=False,
            )
    save(path, material)


if __name__ == "__main__":
    build()
