"""Recount source offsets, visible evidence and overlap directly from both publishers."""

import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from loopqwen.evaluation.extractive import normalized
from loopqwen.io import digest, read, save
from scripts.cross_view_material import SPEC


def grams(text):
    words = re.findall(r"\w+", text.casefold())
    return set(zip(*(words[i:] for i in range(5))))


def check():
    s = read(SPEC)
    panel = read(s["material"])
    train = read(s["training"])
    training = {v["page"]: grams(v["passage"]) for v in train["articles"]}
    questions = {q["question"] for a in train["articles"] for q in a["questions"]}
    with ZipFile("data/research_external/quoref/train-dev.zip") as z:
        source = json.loads(z.read("quoref-train-dev-v0.1/quoref-dev-v0.1.json"))[
            "data"
        ]
    original = {
        q["id"]: (art["title"], p["context"], q)
        for art in source
        for p in art["paragraphs"]
        for q in p["qas"]
    }
    articles = []
    mixed = []
    for g in panel["articles"]:
        set1 = grams(g["passage"])
        nearest, overlap = max(
            ((k, len(set1 & v) / len(set1 | v)) for k, v in training.items()),
            key=lambda v: v[1],
        )
        details = []
        for q in g["questions"]:
            title, passage, orig = original[q["id"]]
            assert (
                title.replace("_", " ") == g["page"]
                and passage == g["passage"]
                and orig["question"] == q["question"]
            )
            assert passage[q["offset"] : q["offset"] + len(q["answer"])] == q["answer"]
            assert q["question"] not in questions
            rows = [w for w in panel["rows"] if w.get("question_id") == q["id"]]
            assert len(rows) == 2 and {w["family"] for w in rows} == {"file", "inline"}
            file = next(w for w in rows if w["family"] == "file")
            inline = next(w for w in rows if w["family"] == "inline")
            assert all(q["question"] in w["questions"][0] for w in rows)
            assert (
                passage in inline["questions"][0]
                and passage not in file["questions"][0]
            )
            arg = file["calls"][0]["arguments"]
            src = list(
                csv.DictReader(io.StringIO(panel["documents"][file["document"]]))
            )
            data = [
                r[arg["value_column"]]
                for r in src
                if r[arg["key_column"]] == arg["key"]
            ]
            assert (
                len(src) == 3
                and data == [passage]
                and file["observation"] == {"value": passage}
            )
            gold = normalized(q["answer"])
            echoed = bool(
                re.search(
                    r"(?<!\w)" + re.escape(gold) + r"(?!\w)", normalized(q["question"])
                )
            )
            details.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "gold": q["answer"],
                    "reference_in_question": echoed,
                    "relative_answer_start": q["offset"] / len(passage),
                    "occurrences_in_passage": passage.count(q["answer"]),
                    "schema": list(src[0]),
                }
            )
            mixed.append(echoed)
        articles.append(
            {
                "page": g["page"],
                "url": g["url"],
                "nearest_training_title": nearest,
                "passage_five_word_jaccard": overlap,
                "questions": details,
            }
        )
    result = {
        "registration_sha256": digest(SPEC),
        "source_sha256": s["source_hashes"],
        "training_articles": len(training),
        "test_articles": len(articles),
        "source_questions": len(mixed),
        "questions_containing_their_answer": sum(mixed),
        "max_passage_five_word_jaccard": max(
            r["passage_five_word_jaccard"] for r in articles
        ),
        "caller_headers": dict(
            Counter(" / ".join(r["questions"][0]["schema"]) for r in articles)
        ),
        "articles": articles,
    }
    output = Path("docs/measurements/cross-view-source-check.json")
    if output.exists():
        assert read(output) == result
    else:
        save(output, result)
    print({k: v for k, v in result.items() if k != "articles"})


if __name__ == "__main__":
    check()
