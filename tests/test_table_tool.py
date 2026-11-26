import sqlite3
from contextlib import closing

import pytest

from loopqwen.evaluation import table_questions
from loopqwen.io import read
from loopqwen.table_tool import database_tool


def test_lookup_uses_bound_values_and_quoted_column_names():
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute('CREATE TABLE "regional sales" ("branch" TEXT,"net ""sales" REAL)')
        db.executemany(
            'INSERT INTO "regional sales" VALUES (?,?)',
            [("West", 2), ("West", 4), ("East", 9)],
        )
        tool = database_tool(db, "regional sales")
        key = tool.parameters["properties"]["column"]["enum"][1]
        condition = {"column": "branch", "operator": "equals", "value": "WEST"}
        assert tool.callback(key, "average", [condition]) == {
            "values": [3.0],
            "truncated": False,
        }
        assert (
            tool.callback(key, "value", [dict(condition, value="' OR 1=1")])["values"]
            == []
        )
        with pytest.raises(ValueError):
            tool.callback("unknown", "value")


def test_publisher_queries_reproduce_both_denotations_and_require_an_observation():
    ops = ["value", "maximum", "minimum", "count", "sum", "average"]
    for row in read("research/data/tool-table-queries-01.json")["rows"]:
        q = row["sql"]
        args = dict(
            column=row["columns"][q["sel"]],
            operation=ops[q["agg"]],
            filters=[
                dict(
                    column=row["columns"][i],
                    operator=["equals", "greater_than", "less_than"][op],
                    value=v,
                )
                for i, op, v in q["conds"]
            ],
        )
        tool = table_questions.tools_for(row)[0]
        obs = tool.callback(**args)
        gold = table_questions.reference(row)
        assert obs["values"] == gold
        tr = dict(status="answered", answer=str(gold[0]), turns=[])
        assert not table_questions.score(row, tr)["success"]
        tr["turns"].append(
            dict(call=dict(name=tool.name, arguments=args), observation=obs)
        )
        assert table_questions.score(row, tr)["success"]


def test_span_pointer_query_preserves_literals_for_live_sqlite():
    from loopqwen.table_spans import actions, arguments

    row = {
        "question": "What is the total amount in North-East for sales above 1,200?",
        "columns": ["region", "sales", "amount"],
        "sql": {"sel": 2, "agg": 4, "conds": [[0, 0, "north-east"], [1, 1, 1200]]},
    }
    selected = actions(row)
    assert selected is not None
    args = arguments(row["question"], row["columns"], 2, 4, selected)
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("create table records(region text, sales real, amount real)")
        db.executemany(
            "insert into records values(?,?,?)",
            [
                ("north-east", 1300, 4),
                ("North-East", 1400, 7),
                ("north-east", 1000, 50),
                ("north-west", 1500, 100),
            ],
        )
        assert database_tool(db, "records").callback(**args)["values"] == [11]
