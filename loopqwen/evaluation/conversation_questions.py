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


def score(row, exchanges, *, literal_prefix=False):
    first, follow = exchanges
    module = row.get("resource") == "module"

    def calls(tr):
        return [t for t in tr["turns"] if "observation" in t]

    left, right = calls(first), calls(follow)
    visible = (first["answer"] or "").lower()
    if row["kind"] == "clarification":
        one = (
            first["status"] == "answered"
            and not left
            and bool(
                re.search(
                    r"module|file|name"
                    if module
                    else r"package|library|distribution|name",
                    visible,
                )
            )
            and bool(re.search(r"\?|which|specify|provide|tell me|need", visible))
        )
    else:

        def valid_listing(turn):
            action = turn["call"]
            if action["name"] != ("list_modules" if module else "list_installed"):
                return False
            prefix = action["arguments"]["prefix"]
            expected = row["prefix"]
            if not literal_prefix:
                prefix, expected = prefix.casefold(), expected.casefold()
            return prefix == expected and turn["observation"] == {
                ("modules" if module else "distributions"): row["matches"]
            }

        one = first["status"] == "answered" and any(valid_listing(t) for t in left)
    value = row["lines"] if module else row["version"]

    def valid_lookup(t):
        action = t["call"]
        if module:
            return action == {
                "name": "module_lines",
                "arguments": {"module": row["target"]},
            } and t["observation"] == {"lines": value}
        return (
            action["name"] == "package_version"
            and canonicalize_name(action["arguments"]["distribution"]) == row["target"]
            and t["observation"] == {"version": value}
        )

    found = any(valid_lookup(t) for t in right)
    correct = (
        follow["status"] == "answered"
        and re.search(
            r"(?<![\w.])" + re.escape(str(value)) + r"(?!\w|\.\d)",
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
