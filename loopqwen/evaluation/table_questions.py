"""Evaluate human WikiSQL questions against the real values in the publisher tables."""

import hashlib
import math
import random
import re
import sqlite3
from contextlib import closing, contextmanager

from ..table_tool import database_tool, quote
from ..tool_session import Tool

INSTRUCTION = (
    "\nUse the database tool. Give just the resulting number as your final answer."
)
SQL_AGG = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
SQL_OP = ["=", ">", "<"]


@contextmanager
def database(row):
    with closing(sqlite3.connect(":memory:")) as connection:
        columns = row["columns"]
        types = {"text": "TEXT", "real": "REAL"}
        fields = ", ".join(
            quote(k) + " " + types[t]
            for k, t in zip(columns, row["types"], strict=True)
        )
        connection.execute("CREATE TABLE records (" + fields + ")")
        connection.executemany(
            "INSERT INTO records VALUES (" + ",".join("?" for _ in columns) + ")",
            row["records"],
        )
        yield connection


def reference(row):
    """Render the publisher's indexed query independently of the exposed Python tool."""
    q, fields = row["sql"], row["columns"]
    field = quote(fields[q["sel"]])
    agg = SQL_AGG[q["agg"]]
    result = f"{agg}({field})" if agg else field
    where = [quote(fields[i]) + " " + SQL_OP[op] + " ?" for i, op, _ in q["conds"]]
    params = [v.lower() if isinstance(v, str) else v for _, _, v in q["conds"]]
    predicate = " WHERE " + " AND ".join(where) if where else ""
    with database(row) as c:
        return [
            r[0]
            for r in c.execute("SELECT " + result + " FROM records" + predicate, params)
        ]


def tools_for(row):
    with database(row) as c:
        tool = database_tool(c, "records")

    def query(**args):
        with database(row) as c:
            return database_tool(c, "records").callback(**args)

    return [Tool(tool.name, tool.description, tool.parameters, query)]


def same(value, gold):
    return isinstance(value, (int, float)) and math.isclose(
        value, gold, rel_tol=1e-5, abs_tol=1e-7
    )


def score(row, trajectory):
    [gold] = reference(row)
    obs = [t["observation"] for t in trajectory["turns"] if "observation" in t]
    path = bool(
        obs
        and obs[-1].get("truncated") is False
        and len(obs[-1]["values"]) == 1
        and same(obs[-1]["values"][0], gold)
    )
    if path:
        # Re-execute the submitted query on a fixed row bootstrap. Equal denotations
        # on the original table alone can reward a query that accidentally selects one value.
        rng = random.Random(hashlib.sha256(row["id"].encode()).hexdigest())
        other = dict(row, records=rng.choices(row["records"], k=len(row["records"])))
        last_call = [t["call"] for t in trajectory["turns"] if "observation" in t][-1]
        with database(other) as db:
            alt = database_tool(db, "records").callback(**last_call["arguments"])
        expected = reference(other)
        path = (
            alt["truncated"] is False
            and len(alt["values"]) == len(expected)
            and all(
                a == b or (isinstance(b, (int, float)) and same(a, b))
                for a, b in zip(alt["values"], expected, strict=True)
            )
        )
    visible = (trajectory["answer"] or "").replace(",", "").strip().rstrip(".")
    matches = re.findall(
        r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", visible
    )
    correct = len(matches) == 1 and same(float(matches[0]), gold)
    complete = trajectory["status"] == "answered"
    return dict(
        success=path and correct and complete,
        path=path,
        correct_answer=correct,
        completed=complete,
    )
