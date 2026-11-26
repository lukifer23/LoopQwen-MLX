"""Compare installed and self-contained local tool entry points in separate processes."""

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from loopqwen.worker import child_lease


def cases(directory):
    catalog = directory / "local prices.json"
    catalog.write_text(
        json.dumps(
            [
                dict(
                    name="read_price",
                    description="Read a recorded item price.",
                    argument="item",
                    records={"notebook": 11, "pencil": 3},
                )
            ]
        )
    )
    db_path = directory / "counts.sqlite"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("CREATE TABLE counts(label TEXT, amount REAL)")
        db.executemany("INSERT INTO counts VALUES(?,?)", [("A", 12), ("B", 31)])
        db.commit()
    return [
        (
            "packages",
            ["--packages"],
            "What version of the Python distribution mlx is installed in this environment? Use the available tool.",
            version("mlx"),
        ),
        (
            "catalog",
            ["--catalog", str(catalog)],
            "What is the recorded price of notebook? Give the number.",
            "11",
        ),
        (
            "calculator",
            ["--calculator"],
            "Use the calculator to multiply 16 by 7. Give the result.",
            "112",
        ),
        (
            "sqlite",
            ["--database", str(db_path), "--table", "counts"],
            "Use the table to find the maximum amount. Give the number.",
            "31",
        ),
    ]


def correct(record):
    kind, value = record["kind"], record["expected"]
    session = record["bundled"]
    calls = [turn for turn in session["turns"] if "observation" in turn]
    if session["status"] != "answered" or len(calls) != 1:
        return False
    action = calls[0]["call"]
    observed = calls[0]["observation"]
    args = action["arguments"]
    grounded = {
        "packages": (
            action["name"] == "package_version"
            and args.get("distribution", "").casefold() == "mlx"
            and observed == {"version": value}
        ),
        "catalog": (
            action["name"] == "read_price"
            and args == {"item": "notebook"}
            and observed == {"value": 11}
        ),
        "calculator": (
            action["name"] == "calculate" and observed.get("value") == "112"
        ),
        "sqlite": (
            action["name"] == "query_table"
            and args.get("operation") == "maximum"
            and args.get("column") == "amount"
            and not args.get("filters")
            and observed == {"values": [31.0], "truncated": False}
        ),
    }[kind]
    return grounded and bool(
        re.search(
            r"(?<![\w.])" + re.escape(value) + r"(?:\.0+)?(?!\w)",
            session["answer"] or "",
        )
    )


def identical(a, b):
    keys = ("prompt_ids", "token_ids", "call", "observation")
    return (a["status"], a["answer"]) == (b["status"], b["answer"]) and [
        [t.get(key) for key in keys] for t in a["turns"]
    ] == [[t.get(key) for key in keys] for t in b["turns"]]


def measure(bundle, budget):
    bundle = bundle.resolve()
    isolated = """import sys,runpy
sys.path.insert(0,sys.argv.pop(1))
runpy.run_module('runtime.tool_cli',run_name='__main__')
assert not any(k=='loopqwen' or k.startswith('loopqwen.') for k in sys.modules)
"""
    kw = child_lease()
    kw["env"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory() as temp:
        records = []
        for kind, flags, prompt, expected in cases(Path(temp)):
            args = [
                "--checkpoint",
                str(bundle),
                "--prompt",
                prompt,
                "--max-tokens",
                "160",
                "--max-turns",
                "3",
                "--trace",
                *flags,
            ]
            record = dict(kind=kind, prompt=prompt, expected=expected)
            for label, cmd in [
                ("installed", [sys.executable, "-m", "loopqwen", "tools"]),
                ("bundled", [sys.executable, "-I", "-c", isolated, str(bundle)]),
            ]:
                budget.check()
                record[label] = json.loads(
                    subprocess.check_output([*cmd, *args], cwd=temp, text=True, **kw)
                )
            record["identical"] = identical(record["installed"], record["bundled"])
            record["correct"] = correct(record)
            records.append(record)
    return records
