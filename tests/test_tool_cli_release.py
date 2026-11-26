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
                    try:
                        expected = fn.callback(**a["call"]["arguments"])
                    except Exception as error:
                        expected = {"error": type(error).__name__}
                    assert a["observation"] == expected
            del assistant
            gc.collect()
            mx.clear_cache()


@pytest.mark.skipif(
    not (BUNDLE / "tool-export-files.json").is_file(),
    reason="Requires the physical two-backbone tool export",
)
def test_small_adapter_overlay_reproduces_changed_public_execution():
    from loopqwen.evaluation import executable_tools
    from loopqwen.evaluation.mechanism import unpack
    from loopqwen.io import read
    from scripts.tool_release import tokens

    joint = Path("research/results/tool-dialogue-counterfactual-01")
    item = next(
        w
        for w in read("research/data/tool-exec-bfcl-01.json")["rows"]
        if w["id"] == "exec_simple_15"
    )
    expected = next(
        r for r in unpack(joint / "candidate.json.gz") if r["id"] == item["id"]
    )
    assistant = Assistant(BUNDLE, tool_adapter=joint)
    result = assistant.respond(
        item["question"][0][0]["content"] + executable_tools.INSTRUCTION,
        executable_tools.tools_for(item),
        max_tokens=384,
        max_turns=3,
    )
    assert tokens(result) == tokens(expected["trajectory"])
    assert executable_tools.score(item, result)["success"]


@pytest.mark.skipif(
    not (BUNDLE / "tool-export-files.json").is_file(),
    reason="Requires the physical two-backbone tool export",
)
def test_followup_reuses_real_observations_without_replaying_callbacks():
    from importlib.metadata import version

    from loopqwen.tool_session import Tool, compact, prompt

    assistant = Assistant(BUNDLE)
    deps, ver = packages()
    calls = []

    def counted(distribution):
        calls.append(distribution)
        return ver.callback(distribution)

    chat = assistant.conversation(
        [deps, Tool(ver.name, ver.description, ver.parameters, counted)]
    )
    question = "What version of the Python distribution mlx is installed in this environment? Use the available tool."
    first = chat.respond(question, max_tokens=256)
    assert first["status"] == "answered" and calls == ["mlx"]
    assert {
        "role": "tool",
        "content": compact({"version": version("mlx")}),
    } in chat.messages
    saved = list(chat.messages)
    chat.messages[-1]["content"] = "changed outside the conversation"
    assert list(chat.messages) == saved
    follow = "Repeat the version from that observation."
    second = chat.respond(follow, max_tokens=96)
    expected = prompt(
        assistant.tokenizer, [*saved, {"role": "user", "content": follow}]
    )
    assert second["turns"][0]["prompt_ids"] == assistant.tokenizer.encode(
        expected, add_special_tokens=False
    )
    assert len(calls) == 1 + sum(
        "observation" in t and t["call"]["name"] == ver.name for t in second["turns"]
    )


@pytest.mark.skipif(
    not (BUNDLE / "tool-export-files.json").is_file(),
    reason="Requires the physical tool export",
)
def test_cli_followup_reuses_real_observation_and_prompt(tmp_path):
    from mlx_lm.utils import load_tokenizer

    from loopqwen.tool_session import initial_messages, prompt

    first = "What version of the Python distribution mlx is installed in this environment? Use the available tool."
    follow = "Repeat the version number from that result."
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "loopqwen",
            "tools",
            "--checkpoint",
            str(BUNDLE),
            "--packages",
            "--prompt",
            first,
            "--follow-up",
            follow,
            "--max-tokens",
            "160",
            "--trace",
        ],
        text=True,
        **child_lease(),
    )
    from importlib.metadata import version

    from loopqwen.tool_session import compact

    tok = load_tokenizer(BUNDLE)
    a, b = json.loads(output)["exchanges"]
    assert a["question"] == first and b["question"] == follow
    history = initial_messages(first, packages())
    observed = []
    for t in a["turns"]:
        if "observation" in t:
            history.extend(
                [
                    {"role": "assistant", "content": t["visible"]},
                    {"role": "tool", "content": compact(t["observation"])},
                ]
            )
            observed.append(t["observation"])
    assert {"version": version("mlx")} in observed
    history.extend(
        [
            {"role": "assistant", "content": a["answer"]},
            {"role": "user", "content": follow},
        ]
    )
    assert b["turns"][0]["prompt_ids"] == tok.encode(
        prompt(tok, history), add_special_tokens=False
    )


@pytest.mark.skipif(
    not (BUNDLE / "tool-export-files.json").is_file(),
    reason="Requires the physical offline assistant and its pinned Base",
)
def test_own_prompt_file_compares_two_real_decoders_on_the_identical_thinking_prompt(
    tmp_path,
):
    from mlx_lm.utils import load_tokenizer

    from loopqwen import REVISION
    from loopqwen.identity import fingerprint
    from loopqwen.model_loader import local_snapshot
    from loopqwen.tool_session import prompt

    question = "A shop opens after the bakery and before the school. Which opens last?"
    source = tmp_path / "question.txt"
    source.write_text(question)
    cmd = [
        sys.executable,
        "-m",
        "loopqwen",
        "ask",
        "--checkpoint",
        str(BUNDLE),
        "--prompt-file",
        str(source),
        "--compare-base",
        "--max-tokens",
        "4",
        "--trace",
    ]
    result = json.loads(subprocess.check_output(cmd, text=True, **child_lease()))
    tok = load_tokenizer(local_snapshot())
    expected = tok.encode(
        prompt(tok, [{"role": "user", "content": question}]), add_special_tokens=False
    )
    assert result["base_revision"] == REVISION and result[
        "conversation_prompt_matches"
    ] == [True]
    for arm in result["runs"].values():
        (row,) = arm
        assert row["initial_sha256"] == fingerprint(expected)
        assert (
            row["turns"][0]["prompt_ids"] == expected and 0 < row["output_tokens"] <= 4
        )
        assert (
            row["request_seconds"] >= row["first_token_seconds"] > 0
            and row["peak_mlx_bytes"] > 0
        )
