"""Use the offline recurrent assistant with caller-selected local tools."""

import argparse
import json
from pathlib import Path


def add_arguments(p, checkpoint=None):
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoint,
        required=checkpoint is None,
        help="Combined language and tool export",
    )
    p.add_argument(
        "--tool-adapter",
        type=Path,
        help="Load a compatible trained tool adapter over the existing frozen backbone",
    )
    p.add_argument(
        "--catalog", type=Path, help="JSON file with read-only lookup definitions"
    )
    p.add_argument(
        "--calculator", action="store_true", help="Expose exact arithmetic expressions"
    )
    p.add_argument(
        "--packages",
        action="store_true",
        help="Expose installed versions and dependencies",
    )
    p.add_argument(
        "--database", type=Path, help="Open an existing SQLite database read-only"
    )
    p.add_argument(
        "--table", help="Name of the one database table exposed to the assistant"
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument(
        "--reuse-prefix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint’s qualified multi-turn prefix policy",
    )
    p.add_argument(
        "--trace", action="store_true", help="Include exact prompt and output tokens"
    )


def respond(args, parser):
    from contextlib import ExitStack, closing

    from .io import read
    from .tool_catalog import load_catalog
    from .tool_model import Assistant

    if bool(args.database) != bool(args.table):
        parser.error("--database and --table must be supplied together")
    with ExitStack() as resources:
        supplied = load_catalog(read(args.catalog)) if args.catalog is not None else []
        if args.calculator:
            from .calculator_tool import tool

            supplied.append(tool())
        if args.packages:
            from .package_tools import tools as packages

            supplied.extend(packages())
        if args.database:
            import sqlite3
            from .table_tool import database_tool

            uri = args.database.resolve().as_uri() + "?mode=ro"
            db = resources.enter_context(closing(sqlite3.connect(uri, uri=True)))
            supplied.append(database_tool(db, args.table))
        if not supplied:
            parser.error("Select a catalog, calculator, packages or a database table")
        result = Assistant(args.checkpoint, tool_adapter=args.tool_adapter).respond(
            args.prompt,
            supplied,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            reuse_prefix=args.reuse_prefix,
        )
    if not args.trace:
        result = {k: v for k, v in result.items() if k != "turns"} | {
            "calls": [
                t["call"] | {"result": t["observation"]}
                for t in result["turns"]
                if "observation" in t
            ]
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, Path("."))
    print(json.dumps(respond(parser.parse_args(), parser), indent=2))
