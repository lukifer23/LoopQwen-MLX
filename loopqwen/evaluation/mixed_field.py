"""Live resources and scorers for a mixed stream sent to the public assistant."""

import hashlib
import re
from pathlib import Path
from packaging.utils import canonicalize_name

from ..package_tools import tools as package_tools
from ..tool_session import Tool
from . import program_questions, table_questions


def catalog(row, material, replay=False):
    kind = row["family"]
    if kind == "reading":
        return []
    if kind == "arithmetic":
        return program_questions.tools_for(row)
    if kind == "database":
        return table_questions.tools_for(row)
    if kind == "package":
        return package_tools(material["environment"] if replay else None)
    if kind != "file":
        raise ValueError("Unknown field task")

    def stat(path, field):
        expected = material["files"].get(path)
        if expected is None:
            raise FileNotFoundError(path)
        if replay:
            return {field: expected[field]}
        actual = Path(path).read_bytes()
        if hashlib.sha256(actual).hexdigest() != expected["sha256"]:
            raise ValueError("Project file changed")
        return {field: len(actual) if field == "bytes" else len(actual.splitlines())}

    schema = dict(
        type="object",
        properties=dict(path=dict(type="string")),
        required=["path"],
        additionalProperties=False,
    )
    return [
        Tool(
            "file_bytes",
            "Get the size in bytes of a project file by its exact repository-relative path.",
            schema,
            lambda path: stat(path, "bytes"),
        ),
        Tool(
            "file_lines",
            "Get the number of lines in a project file by its exact repository-relative path.",
            schema,
            lambda path: stat(path, "lines"),
        ),
    ]


def grade(row, tr, tok):
    kind = row["family"]
    done = tr["status"] == "answered"
    answer = tr["answer"] or ""
    if kind == "reading":
        from ..research.bridge_reading import grade as read_grade

        metric = read_grade(tok, tr["turns"][-1]["token_ids"], row["gold"])
        return dict(
            success=metric["content"],
            answer=metric["content"],
            grounded=None,
            completed=done,
        )
    if kind in ("arithmetic", "database"):
        scorer = program_questions if kind == "arithmetic" else table_questions
        val = scorer.score(row, tr)
        return dict(
            success=val["correct_answer"] if kind == "arithmetic" else val["success"],
            answer=val["correct_answer"],
            grounded=val["success"],
            completed=done,
        )
    turns = [t for t in tr["turns"] if "observation" in t]
    path = False
    if kind == "package":
        correct = bool(
            re.search(r"(?<![\w.])" + re.escape(row["gold"]) + r"(?!\w|\.\d)", answer)
        )
        if len(turns) == 1:
            c = turns[0]["call"]
            path = (
                c["name"] == "package_version"
                and canonicalize_name(c["arguments"].get("distribution", ""))
                == canonicalize_name(row["name"])
                and turns[0]["observation"] == {"version": row["gold"]}
            )
    else:
        correct = bool(
            re.search(
                r"(?<![\w.])" + re.escape(row["gold"]) + r"(?!\w|\.\d)",
                answer.replace(",", ""),
            )
        )
        if len(turns) == 1:
            t = turns[0]
            path = t["call"] == dict(
                name="file_" + row["unit"], arguments={"path": row["path"]}
            ) and t["observation"] == {row["unit"]: int(row["gold"])}
    return dict(
        success=path and correct and done,
        answer=correct and done,
        grounded=path and correct and done,
        completed=done,
    )
