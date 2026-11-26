"""Build a human query-program training partition with executable span targets and fresh tables."""

import hashlib
import json
import shutil
import sqlite3
import tarfile
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from paper_overlap import normalize, shingle
from table_question_panel import REV, SHA, select

from loopqwen.evaluation.table_questions import SQL_AGG, SQL_OP, reference
from loopqwen.io import digest, read, save
from loopqwen.table_spans import COMPARISONS, actions, arguments, words

DATA = Path("research/data")


def canon(value):
    text = str(value).strip().casefold()
    try:
        return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return text


def freeze():
    earlier = read(DATA / "tool-table-queries-01.json")["rows"]
    earlier += [
        r
        for r in read(DATA / "tool-sql-learning-panel.json")["rows"]
        if r["provider"] == "wikisql"
    ]
    mixed = read(read("research/experiments/release-mixed-field-01.json")["material"])
    earlier += [w for w in mixed["rows"] if w["family"] == "database"]
    forbidden = {w["table_id"] for w in earlier}
    forbidden_questions = {w["question"].strip().casefold() for w in earlier}
    test = select("test", 8, "schema-parser-test-80219", forbidden, forbidden_questions)
    forbidden.update(w["table_id"] for w in test)
    forbidden_questions.update(w["question"].strip().casefold() for w in test)
    held_words = {normalize(w["question"], True) for w in earlier + test}
    held_grams = [shingle(words_) for words_ in held_words]
    src = Path("data/research_external/wikisql")
    tar = src / "source.tar.bz2"
    if digest(tar) != SHA:
        raise ValueError("Different publisher data")
    with tarfile.open(tar, "r:bz2") as stream:
        db_path = src / "train.db"
        expected = hashlib.file_digest(
            stream.extractfile("data/train.db"), "sha256"
        ).hexdigest()
        if not db_path.exists() or digest(db_path) != expected:
            temporary = db_path.with_suffix(".partial")
            with temporary.open("wb") as f:
                shutil.copyfileobj(stream.extractfile("data/train.db"), f)
            if digest(temporary) != expected:
                raise ValueError("Incomplete WikiSQL training database")
            temporary.replace(db_path)
        queries = [json.loads(line) for line in stream.extractfile("data/train.jsonl")]
        tables = {
            t["id"]: t
            for t in map(json.loads, stream.extractfile("data/train.tables.jsonl"))
        }
    order = sorted(
        range(len(queries)),
        key=lambda i: hashlib.sha256(f"tbl-parser-fit:{i}".encode()).digest(),
    )
    counts = Counter()
    train = []
    with closing(
        sqlite3.connect((src / "train.db").resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        for idx in order:
            w = queries[idx]
            plan = w["sql"]
            agg = plan["agg"]
            table = w["table_id"]
            question = w["question"]
            if (
                counts[agg] >= 384
                or table in forbidden
                or question.strip().casefold() in forbidden_questions
            ):
                continue
            cols = tables[table]["header"]
            if not (
                2 <= len(cols) <= 24
                and len(set(c.casefold() for c in cols)) == len(cols)
                and 1 <= len(words(question)) <= 64
            ):
                continue
            token_set = normalize(question, True)
            grams = shingle(token_set)
            if token_set in held_words or (
                grams
                and any(
                    len(grams & ref) >= 0.6 * len(grams | ref)
                    for ref in held_grams
                    if ref
                )
            ):
                continue
            row = {
                "id": f"wikisql-train-{idx}",
                "table_id": table,
                "columns": cols,
                "question": question,
                "sql": plan,
            }
            cond = actions(row)
            if cond is None:
                continue
            pred = arguments(question, cols, plan["sel"], agg, cond)
            values = sorted(
                (c["column"], c["operator"], canon(c["value"])) for c in pred["filters"]
            )
            gold = sorted(
                (cols[i], COMPARISONS[op], canon(v)) for i, op, v in plan["conds"]
            )
            if values != gold:
                continue
            # Both programs are executed on the publisher database. Actual
            # literals and column binding must preserve the reference result.
            pieces = [f"col{i} COLLATE NOCASE {SQL_OP[op]} ?" for i, op, _, _ in cond]
            if not all(isinstance(v[2], (int, float, str)) for v in plan["conds"]):
                continue
            projected = f"col{plan['sel']}"
            if SQL_AGG[agg]:
                projected = f"{SQL_AGG[agg]}({projected})"
            stmt = f'SELECT {projected} FROM "table_{table.replace("-", "_")}"'
            bind = lambda keys: stmt + (" WHERE " + " AND ".join(keys) if keys else "")
            refs = [
                f"col{i} COLLATE NOCASE {SQL_OP[op]} ?" for i, op, _ in plan["conds"]
            ]
            expected = db.execute(
                bind(refs), [v for _, _, v in plan["conds"]]
            ).fetchall()
            got = db.execute(
                bind(pieces), [f["value"] for f in pred["filters"]]
            ).fetchall()
            if not expected or all(r[0] is None for r in expected) or got != expected:
                continue
            row["conditions"] = cond
            train.append(row)
            forbidden.add(table)
            forbidden_questions.add(question.strip().casefold())
            counts[agg] += 1
            if min((counts[i] for i in range(6))) == 384:
                break
    if len(train) != 2304:
        raise ValueError(f"Training strata {counts}")
    save(
        DATA / "table-parser-training.json",
        {"source": REV, "archive_sha256": SHA, "rows": train},
    )
    for r in test:
        assert len(reference(r)) == 1
    save(DATA / "table-parser-panel.json", {"source": REV, "rows": test})
    print(len(train), len(test), counts, flush=True)


if __name__ == "__main__":
    freeze()
