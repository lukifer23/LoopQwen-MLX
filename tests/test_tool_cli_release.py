"""Qualify the installed command against the public Python API on the same native export."""

import gc
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import pytest

from loopqwen.calculator_tool import tool as calculator
from loopqwen.package_tools import tools as packages
from loopqwen.table_tool import database_tool
from loopqwen.tool_catalog import load_catalog
from loopqwen.tool_model import Assistant
from loopqwen.worker import child_lease

BUNDLE = Path("exports/tool-language-01")


@pytest.mark.skipif(
    not (BUNDLE / "tool-export-files.json").is_file(),
    reason="Requires the physical two-backbone tool export",
)
def test_real_tool_cli_matches_public_api(tmp_path):
    defs = [
        {
            "name": "read_price",
            "description": "Read a recorded item price.",
            "argument": "item",
            "records": {"notebook": 11, "pencil": 3},
        }
    ]
    catalog = tmp_path / "local-prices.json"
    catalog.write_text(json.dumps(defs))
    database = tmp_path / "labels and counts.sqlite"
    with closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE counts(label TEXT, amount REAL)")
        db.executemany("INSERT INTO counts VALUES(?,?)", [("A", 12), ("B", 31)])
        db.commit()
        cases = [
            (
                ["--packages"],
                packages(),
                "What version of the Python distribution mlx is installed in this environment? Use the available tool.",
            ),
            (
                ["--catalog", str(catalog)],
                load_catalog(defs),
                "What is the recorded price of notebook? Give the number.",
            ),
            (
                ["--calculator"],
                [calculator()],
                "Use the calculator to multiply 16 by 7. Give the result.",
            ),
            (
                ["--database", str(database), "--table", "counts"],
                [database_tool(db, "counts")],
                "Use the table to find the maximum amount. Give the number.",
            ),
        ]
        for flags, tools, prompt in cases:
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "loopqwen",
                    "tools",
                    "--checkpoint",
                    str(BUNDLE),
                    "--prompt",
                    prompt,
                    "--max-tokens",
                    "160",
                    "--max-turns",
                    "3",
                    "--trace",
                    *flags,
                ],
                text=True,
                **child_lease(),
            )
            cli = json.loads(output)
            assistant = Assistant(BUNDLE)
            public = assistant.respond(prompt, tools, max_tokens=160, max_turns=3)
            assert (cli["status"], cli["answer"]) == (
                public["status"],
                public["answer"],
            )
            for a, b in zip(cli["turns"], public["turns"], strict=True):
                for key in ("prompt_ids", "token_ids", "call", "observation"):
                    assert a.get(key) == b.get(key)
                if "observation" in a:
                    fn = next(t for t in tools if t.name == a["call"]["name"])
                    assert a["observation"] == fn.callback(**a["call"]["arguments"])
            del assistant
            gc.collect()
            mx.clear_cache()
