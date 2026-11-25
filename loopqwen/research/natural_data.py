"""Ground reader training in multiple real paragraphs from each publisher training article."""

import re
from pathlib import Path
from random import Random

from ..identity import fingerprint
from ..io import digest, read
from . import core


def make(t, s):
    source = Path("data/research_external/squad-train-v1.1.json")
    if digest(source) != s["source_sha256"]:
        raise ValueError("SQuAD training source differs")
    train = read(source)["data"]
    known = set()
    for path in s["exclude_training"]:
        known.update(read(path)["ordered_questions"])
    train = [
        a for a in train if len(a["paragraphs"]) >= 5 and a["title"] not in s["titles"]
    ]
    entries = [
        (a, pi, q)
        for a in train
        for pi, p in enumerate(a["paragraphs"])
        for q in p["qas"]
        if q["id"] not in known
    ]
    entries.sort(key=lambda e: fingerprint([s["seed"], e[2]["id"]]))
    rows, docs = [], []
    for a, pi, q in entries:
        paragraphs = [p["context"] for p in a["paragraphs"]]
        ann = q["answers"][0]
        start = ann["answer_start"]
        ans = ann["text"]
        para = paragraphs[pi]
        if para[start : start + len(ans)] != ans or len(para) > 2200:
            continue

        def prompt(body):
            question = (
                body
                + "\n\nQuestion: "
                + q["question"]
                + "\nReason briefly. Give just the shortest phrase from the article that answers the question."
            )
            return t.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

        r = Random(int(fingerprint([s["seed"], q["id"]])[:12], 16))
        others = [
            i
            for i, p in enumerate(paragraphs)
            if i != pi and p != para and len(p) < 2200
        ]
        if len(others) < 3:
            continue
        selected = r.sample(others, 3) + [pi]
        r.shuffle(selected)
        body = "\n\n".join(paragraphs[i] for i in selected)
        text = prompt(body)
        ids = t.encode(text, add_special_tokens=False)
        if not 320 <= len(ids) <= 1232:
            continue
        if para not in text:
            continue
        char = text.index(para) + start
        lower = len(t.encode(text[:char], add_special_tokens=False))
        upper = len(t.encode(text[: char + len(ans)], add_special_tokens=False))
        sentence = (
            re.split(r"(?<=[.?!])\s+", para[:start])[-1]
            + ans
            + re.split(r"(?<=[.?!])\s+", para[start + len(ans) :])[0]
        )
        if len(docs) < s["updates"] * s["logical_batch"]:
            doc = core.doc(t, prompt(para), sentence + "\n</think>\n\n" + ans)
            if len(doc["ids"]) - 1 <= s["cap_positions"]:
                docs.append(doc)
        rows.append(
            {
                "id": q["id"],
                "ids": ids,
                "answer_start": lower,
                "answer_stop": max(lower + 1, upper),
            }
        )
        if len(rows) == s["passages"]:
            break
    if len(rows) != s["passages"] or len(docs) != s["updates"] * s["logical_batch"]:
        raise ValueError("Insufficient natural evidence training material")
    return rows, docs
