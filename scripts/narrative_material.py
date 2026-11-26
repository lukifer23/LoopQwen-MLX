"""Freeze one entire public-domain novel and a consistently renamed counterfactual."""

import csv
import hashlib
import io
import re
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer
from loopqwen.identity import fingerprint
from loopqwen.io import save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt

BOOK = "https://raw.githubusercontent.com/GITenberg/McTeague--A-Story-of-San-Francisco_165/6e785cde32b3445e271c91bcc690c9f479ad0817/165.txt"
RELEASE = "https://raw.githubusercontent.com/google-deepmind/narrativeqa/904246f6d1fe99a99a08a03501fe3e619af2cee5/"
QUESTION = "Who informs City Hall that McTeague does not in fact have a dental license?"


def main():
    raw = urlopen(BOOK, timeout=60).read()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    marks = list(
        re.finditer(
            r"(?im)^\*\*\* *(?:START|END) OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
            source,
        )
    )
    if len(marks) != 2:
        raise ValueError("Missing book boundaries")
    body = source[marks[0].end() : marks[1].start()].strip()
    qa = urlopen(RELEASE + "qaps.csv", timeout=60).read()
    rows = [
        r
        for r in csv.DictReader(io.StringIO(qa.decode()))
        if r["set"] == "test" and r["question"] == QUESTION
    ]
    if len(rows) != 1 or {
        rows[0][k].strip(". ").lower() for k in ("answer1", "answer2")
    } != {"marcus"}:
        raise ValueError("Different publisher answer")
    changes = {"Marcus": "Felix", "Mark": "Ned", "Schouler": "Radford"}
    altered = body
    counts = {}
    for old, new in changes.items():
        if re.search(r"\b" + new + r"\b", body, re.I):
            raise ValueError("Existing alternate name")
        altered, counts[old] = re.subn(r"\b" + old + r"\b", new, altered)
    tok = load_tokenizer(local_snapshot())
    panel = []
    for variant, text, answer in [
        ("original", body, "Marcus"),
        ("renamed", altered, "Felix"),
    ]:
        question = (
            "Read the complete book below. Answer the question with the person’s name.\n\n"
            + text
            + "\n\nQuestion: "
            + QUESTION
        )
        ids = tok.encode(
            prompt(tok, [dict(role="user", content=question)]), add_special_tokens=False
        )
        if not 131072 <= len(ids) <= 180000:
            raise ValueError("Whole novel outside enrolled lengths")
        panel.append(
            dict(id=variant, answer=answer, prompt_ids=ids, sha256=fingerprint(ids))
        )
    save(
        "research/data/narrative-full-book-01.json",
        dict(
            title="McTeague, A Story of San Francisco",
            author="Frank Norris",
            source=BOOK,
            book_sha256=hashlib.sha256(raw).hexdigest(),
            qa=RELEASE + "qaps.csv",
            qa_sha256=hashlib.sha256(qa).hexdigest(),
            publisher_row=rows[0],
            changed_names=changes,
            substitutions=counts,
            panel=panel,
        ),
    )
    print([(r["id"], len(r["prompt_ids"])) for r in panel], counts)


if __name__ == "__main__":
    main()
