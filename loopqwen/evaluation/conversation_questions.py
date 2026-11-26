"""Live package inventory and endpoint scoring for two-user-turn requests."""

import re
from importlib.metadata import distributions

from packaging.utils import canonicalize_name

from ..package_tools import tools as package_tools
from ..tool_session import Tool


def catalog(environment=None):
    def list_installed(prefix):
        names = (
            environment.keys()
            if environment is not None
            else (
                canonicalize_name(d.metadata["Name"])
                for d in distributions()
                if d.metadata.get("Name")
            )
        )
        return {
            "distributions": sorted(
                {name for name in names if name.startswith(prefix.casefold())}
            )
        }

    return [
        Tool(
            "list_installed",
            "List the installed Python distributions whose names start with a supplied prefix, sorted alphabetically.",
            {
                "type": "object",
                "properties": {"prefix": {"type": "string"}},
                "required": ["prefix"],
                "additionalProperties": False,
            },
            list_installed,
        ),
        package_tools(environment)[1],
    ]


def score(row, exchanges):
    first, follow = exchanges

    def calls(tr):
        return [t for t in tr["turns"] if "observation" in t]

    left, right = calls(first), calls(follow)
    visible = (first["answer"] or "").lower()
    if row["kind"] == "clarification":
        one = (
            first["status"] == "answered"
            and not left
            and bool(re.search(r"package|library|distribution|name", visible))
            and bool(re.search(r"\?|which|specify|provide|tell me|need", visible))
        )
    else:
        one = first["status"] == "answered" and any(
            t["call"]
            == {"name": "list_installed", "arguments": {"prefix": row["prefix"]}}
            and t["observation"] == {"distributions": row["matches"]}
            for t in left
        )
    found = any(
        t["call"]["name"] == "package_version"
        and canonicalize_name(t["call"]["arguments"].get("distribution", ""))
        == row["target"]
        and t["observation"] == {"version": row["version"]}
        for t in right
    )
    correct = (
        follow["status"] == "answered"
        and re.search(
            r"(?<![\w.])" + re.escape(row["version"]) + r"(?!\w|\.\d)",
            follow["answer"] or "",
        )
        is not None
    )
    return {
        "first": one,
        "grounded": found,
        "content": correct,
        "success": one and found and correct,
    }
