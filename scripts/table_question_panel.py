"""Freeze one WikiSQL test question per table, stratified by the six publisher operations."""

import hashlib
import json
import math
import shutil
import sqlite3
import tarfile
from contextlib import closing
from pathlib import Path
from urllib.request import urlopen

from loopqwen.evaluation.table_questions import SQL_AGG, SQL_OP, reference
from loopqwen.io import digest, save

REV = "cffb423077756d04c1bac5bcd45167c86903fbcb"
URL = "https://raw.githubusercontent.com/salesforce/WikiSQL/" + REV + "/data.tar.bz2"
SHA = "755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881"


def make():
    root = Path("data/research_external/wikisql")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.tar.bz2"
    if not source.exists():
        with urlopen(URL, timeout=50) as response, source.open("wb") as f:
            shutil.copyfileobj(response, f)
    assert digest(source) == SHA
    with tarfile.open(source, "r:bz2") as t:
        queries = [json.loads(line) for line in t.extractfile("data/test.jsonl")]
        tables = {
            r["id"]: r for r in map(json.loads, t.extractfile("data/test.tables.jsonl"))
        }
        with (root / "test.db").open("wb") as f:
            shutil.copyfileobj(t.extractfile("data/test.db"), f)
    selected = []
    seen = set()
    counts = [0] * 6
    keyed = sorted(
        enumerate(queries),
        key=lambda u: hashlib.sha256(
            ("sqlite-transfer-6905:" + str(u[0])).encode()
        ).digest(),
    )
    with closing(
        sqlite3.connect((root / "test.db").resolve().as_uri() + "?mode=ro", uri=True)
    ) as c:
        for i, q in keyed:
            name, sql = q["table_id"], q["sql"]
            k = sql["agg"]
            if (
                counts[k] == 4
                or name in seen
                or not sql["conds"]
                or any(op not in range(3) for _, op, _ in sql["conds"])
            ):
                continue
            table = tables[name]
            cols = table["header"]
            if (
                len({s.casefold() for s in cols}) != len(cols)
                or any(not s or len(s) > 65 for s in cols)
                or len(cols) > 12
            ):
                continue
            sql_table = "table_" + name.replace("-", "_")
            if not all(ch.isalnum() or ch == "_" for ch in sql_table):
                raise ValueError("Unrecognized publisher table name")
            values = c.execute('SELECT * FROM "' + sql_table + '"').fetchall()
            if not 4 <= len(values) <= 64:
                continue
            # Verify that a native SQLite query against the publisher's database has
            # the same numeric answer as our separately reconstructed named-column table.
            projection = "col" + str(sql["sel"])
            if k:
                projection = f"{SQL_AGG[k]}({projection})"
            where = " AND ".join(
                "col" + str(ix) + " " + SQL_OP[op] + " ?" for ix, op, _ in sql["conds"]
            )
            params = []
            for ix, _, v in sql["conds"]:
                if table["types"][ix] == "real" and isinstance(v, str):
                    try:
                        v = float(v.replace(",", ""))
                    except ValueError:
                        break
                params.append(v.lower() if isinstance(v, str) else v)
            if len(params) != len(sql["conds"]):
                continue
            official = c.execute(
                "SELECT " + projection + ' FROM "' + sql_table + '" WHERE ' + where,
                params,
            ).fetchall()
            if (
                len(official) != 1
                or not isinstance(official[0][0], (int, float))
                or not math.isfinite(official[0][0])
                or official[0][0] == 0
            ):
                continue
            row = dict(
                id="wikisql-test-" + str(i),
                table_id=name,
                question=q["question"],
                columns=cols,
                types=table["types"],
                records=values,
                sql=sql,
            )
            if reference(row) != [official[0][0]]:
                continue
            seen.add(name)
            counts[k] += 1
            selected.append(row)
            if len(selected) == 24:
                break
    assert counts == [4] * 6
    save(
        "research/data/tool-table-queries-01.json",
        dict(source=URL, sha256=SHA, rows=selected),
    )
    print(len(selected), "distinct test tables; published aggregation counts", counts)


if __name__ == "__main__":
    make()
