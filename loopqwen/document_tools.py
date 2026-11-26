"""Full-text search over snapshots of UTF-8 documents explicitly supplied by the caller."""

import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from .tool_session import Tool


def chunks(text):
    for paragraph in re.finditer(r"\S[\s\S]*?(?=\n[ \t\r]*\n|\Z)", text):
        tokens = list(re.finditer(r"\S+", paragraph.group()))
        for start in range(0, len(tokens), 72):
            end = min(start + 96, len(tokens))
            lo = paragraph.start() + tokens[start].start()
            hi = paragraph.start() + tokens[end - 1].end()
            yield lo, hi, text[lo:hi]
            if end == len(tokens):
                break


@contextmanager
def documents(paths):
    names = []
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "body, document UNINDEXED, start UNINDEXED, stop UNINDEXED, tokenize='porter unicode61')"
        )
        for path in map(Path, paths):
            if path.name in names:
                raise ValueError("Document names must be distinct")
            names.append(path.name)
            text = path.read_bytes().decode("utf-8-sig")
            db.executemany(
                "INSERT INTO passages(body,document,start,stop) VALUES(?,?,?,?)",
                ((body, path.name, lo, hi) for lo, hi, body in chunks(text)),
            )
        if not names:
            raise ValueError("Supply at least one document")
        db.commit()

        def search(query, document=None):
            if document is not None and document not in names:
                raise ValueError("Choose one of the supplied documents")
            words = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))
            if not words:
                raise ValueError("Search needs at least one word")
            match = " OR ".join(f'"{w}"' for w in words)
            rows = db.execute(
                "SELECT document, start, stop, body FROM passages WHERE passages MATCH ? "
                "AND (? IS NULL OR document = ?) ORDER BY rank, rowid LIMIT 3",
                (match, document, document),
            )
            return {
                "passages": [
                    {
                        "document": file,
                        "start": int(lo),
                        "stop": int(hi),
                        "text": excerpt,
                    }
                    for file, lo, hi, excerpt in rows
                ]
            }

        yield [
            Tool(
                "search_documents",
                "Search the supplied documents for relevant words and return up to three original excerpts. Optionally name one file.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "document": {"type": "string", "enum": names},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                search,
            )
        ]
