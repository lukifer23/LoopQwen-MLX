"""Independently execute held-out published SQL on the original SQLite file."""

import csv
import hashlib
import io
import json
import re
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from loopqwen.io import digest, read, save

PANEL = Path("research/data/role-binding-panel.json")
DEST = Path("docs/measurements/role-binding-source-check.json")


def canonical(value):
    s = str(value).strip().casefold()
    try:
        return str(Fraction(s.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return s


def run():
    panel = read(PANEL)
    source = Path("data/research_external/wikisql/source.tar.bz2")
    assert digest(source) == panel["sources"]["wikisql"]["sha256"]
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(source, "r:bz2") as z:
        db = Path(tmp) / "test.db"
        raw = z.extractfile("data/test.db").read()
        db.write_bytes(raw)
        original = list(map(json.loads, z.extractfile("data/test.jsonl")))
        tables = {
            v["id"]: v for v in map(json.loads, z.extractfile("data/test.tables.jsonl"))
        }
        checks = []
        with closing(
            sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            for w in panel["rows"]:
                if w["family"] != "table":
                    continue
                source_row = original[int(w["id"].rsplit("-", 1)[1])]
                table = tables[source_row["table_id"]]
                assert [source_row["question"]] == w["questions"] and table["id"] == w[
                    "cluster"
                ]
                gold_sql = source_row["sql"]
                ((key, op, val),) = gold_sql["conds"]
                target = gold_sql["sel"]
                assert op == 0 and gold_sql["agg"] == 0
                identifier = "table_" + table["id"].replace("-", "_")
                assert (
                    re.fullmatch(r"table_[0-9_]+", identifier)
                    and type(key) is type(target) is int
                )
                stmt = f'SELECT col{target} FROM "{identifier}" WHERE col{key} = ? COLLATE NOCASE'
                values = [r[0] for r in connection.execute(stmt, (val,))]
                csv_rows = list(
                    csv.DictReader(io.StringIO(panel["documents"][w["document"]]))
                )
                csv_values = [
                    r[table["header"][target]]
                    for r in csv_rows
                    if r[table["header"][key]] == w["arguments"]["key"]
                ]
                assert csv_values == [w["gold"]]
                checks.append(
                    {
                        "id": w["id"],
                        "question": source_row["question"],
                        "sql": stmt,
                        "literal": val,
                        "official_sqlite_values": values,
                        "original_table_text": csv_values,
                        "canonical_agreement": list(map(canonical, values))
                        == list(map(canonical, csv_values)),
                    }
                )
    result = {
        "panel_sha256": digest(PANEL),
        "publisher_db_sha256": hashlib.sha256(raw).hexdigest(),
        "agree": sum(c["canonical_agreement"] for c in checks),
        "questions": checks,
    }
    if DEST.exists():
        assert read(DEST) == result
    else:
        save(DEST, result)
    print(result["agree"], "of", len(checks), "original SQLite denotations agree")
    for c in checks:
        if not c["canonical_agreement"]:
            print(c)


if __name__ == "__main__":
    run()
