"""Ask the recurrent assistant about packages installed in the active Python environment."""

import argparse
from importlib.metadata import PackageNotFoundError, metadata, version

from loopqwen.tool_model import Assistant
from loopqwen.tool_session import Tool


def package_version(distribution):
    try:
        return {"version": version(distribution)}
    except PackageNotFoundError:
        return {"error": "package is not installed"}


def package_summary(distribution):
    try:
        return {"summary": metadata(distribution).get("Summary", "")}
    except PackageNotFoundError:
        return {"error": "package is not installed"}


def tools():
    params = {
        "type": "object",
        "properties": {
            "distribution": {
                "type": "string",
                "description": "Installed Python distribution name",
            }
        },
        "required": ["distribution"],
        "additionalProperties": False,
    }
    return [
        Tool(
            "package_version",
            "Read the installed version of a Python distribution.",
            params,
            package_version,
        ),
        Tool(
            "package_summary",
            "Read the publisher's one-line description of a Python distribution.",
            params,
            package_summary,
        ),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("question")
    args = parser.parse_args()
    result = Assistant(args.checkpoint).respond(args.question, tools(), max_tokens=256)
    for turn in result["turns"]:
        if "observation" in turn:
            print(turn["call"], "=>", turn["observation"])
    print(result["answer"] if result["answer"] is not None else result["status"])
