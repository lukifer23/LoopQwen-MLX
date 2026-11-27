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
        "--tool-format",
        choices=("compact", "qwen"),
        default="compact",
        help="Use compact JSON or the official Qwen function-call template",
    )
    policies = p.add_mutually_exclusive_group()
    policies.add_argument(
        "--tool-adapter",
        type=Path,
        help="Load a compatible trained tool adapter over the existing frozen backbone",
    )
    policies.add_argument(
        "--session-policy",
        type=Path,
        help="Use compatible reasoning and action adapters sharing one resident backbone",
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
    p.add_argument(
        "--resource",
        action="append",
        choices=("calendar", "zone", "unicode", "distribution"),
        default=[],
        help="Expose both read-only properties of a local resource. Can be repeated.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Expose a UTF-8 CSV file for read-only row counts and cell lookups. Can be repeated.",
    )
    p.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="Expose the byte size and line count of this file. Can be repeated.",
    )
    p.add_argument(
        "--document",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Allow passage search over this UTF-8 text or Markdown file. Can be repeated.",
    )
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the complete UTF-8 user request from disk",
    )
    p.add_argument(
        "--follow-up",
        action="append",
        default=[],
        metavar="TEXT",
        help="Send another user message in the same live conversation. Can be repeated.",
    )
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument(
        "--reuse-prefix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint’s qualified multi-turn prefix policy",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Include exact prompt and output tokens and resident attention-site histories",
    )


def open_assistant(args):
    if getattr(args, "session_policy", None) is not None:
        if getattr(args, "tool_format", "compact") != "compact":
            raise ValueError(
                "This session policy was fitted with the compact tool template"
            )
        from .io import read

        policy = read(args.session_policy / "registration.json")
        if policy.get("callback_recovery"):
            from .callback_recovery import RecoveryAssistant

            return RecoveryAssistant(args.checkpoint, args.session_policy)
        if policy.get("unified"):
            from .unified_session import UnifiedAssistant

            return UnifiedAssistant(args.checkpoint, args.session_policy)
        from .shared_session import SharedAssistant

        return SharedAssistant(args.checkpoint, args.session_policy)
    from .tool_model import Assistant

    return Assistant(
        args.checkpoint,
        tool_adapter=args.tool_adapter,
        tool_format=getattr(args, "tool_format", "compact"),
    )


def respond(args, parser):
    from contextlib import ExitStack, closing

    from .io import read
    from .tool_catalog import load_catalog

    if getattr(args, "prompt_file", None) is not None:
        args.prompt = args.prompt_file.read_bytes().decode("utf-8")
    if bool(args.database) != bool(args.table):
        parser.error("--database and --table must be supplied together")
    with ExitStack() as resources:
        supplied = load_catalog(read(args.catalog)) if args.catalog is not None else []
        if getattr(args, "csv", None):
            from .csv_tools import tools as csv_tools

            supplied.extend(csv_tools(args.csv))
        if getattr(args, "file", None):
            from .file_tools import tools as file_tools

            supplied.extend(file_tools(args.file))
        if getattr(args, "document", None):
            from .document_tools import documents

            supplied.extend(resources.enter_context(documents(args.document)))
        if args.calculator:
            from .calculator_tool import tool

            supplied.append(tool())
        if args.packages:
            from .package_tools import tools as packages

            supplied.extend(packages())
        if args.resource:
            from .resource_tools import tools as local_tools

            for family in dict.fromkeys(args.resource):
                supplied.extend(local_tools(family))
        if args.database:
            import sqlite3

            from .table_tool import database_tool

            uri = args.database.resolve().as_uri() + "?mode=ro"
            db = resources.enter_context(closing(sqlite3.connect(uri, uri=True)))
            if getattr(args, "compare_base", False):
                db.execute("BEGIN")
            supplied.append(database_tool(db, args.table))
        if not supplied and not getattr(args, "allow_plain", False):
            parser.error(
                "Select a file, document, CSV, resource, catalog, calculator, packages or a database table"
            )
        options = {
            "max_tokens": args.max_tokens,
            "max_turns": args.max_turns,
            "reuse_prefix": args.reuse_prefix,
        }
        if args.trace:
            options["record_kv_history"] = True
        prompts = [args.prompt, *args.follow_up]
        if any(not text.strip() for text in prompts):
            parser.error("Every conversation message needs text")
        if getattr(args, "compare_base", False):
            from .tool_compare import compare

            return compare(args, supplied, options)
        assistant = open_assistant(args)
        if args.follow_up:
            chat = assistant.conversation(supplied)
            return {
                "exchanges": [
                    {
                        "question": text,
                        **present(chat.respond(text, **options), args.trace),
                    }
                    for text in prompts
                ]
            }
        return present(assistant.respond(args.prompt, supplied, **options), args.trace)


def present(result, trace):
    if trace:
        return result
    calls = []
    for t in result["turns"]:
        if "observation" in t:
            calls.append({**t["call"], "result": t["observation"]})
        elif "observations" in t:
            calls.extend(
                {**a, "result": v}
                for a, v in zip(t["calls"], t["observations"], strict=True)
            )
    return {k: v for k, v in result.items() if k != "turns"} | {"calls": calls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, Path("."))
    print(json.dumps(respond(parser.parse_args(), parser), indent=2))
